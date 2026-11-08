"""Session-local grading data, exact lookup tools and Excel export."""
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import hashlib
import re
import tempfile
import threading
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from langchain_core.tools import tool
from parse_exam import parse_key, parse_students, parse_questions, grade

# PDFium is not thread-safe. Only parsing is serialized, never chatbot calls.
PARSE_LOCK = threading.RLock()

def fingerprint(files):
    digest = hashlib.sha256()
    for content in files:
        digest.update(len(content).to_bytes(8, 'big')); digest.update(content)
    return digest.hexdigest()

@dataclass
class ExamData:
    exam_id: str
    key: dict
    students: list
    questions: dict
    images: dict
    details: list
    scores: list
    stats: dict
    metadata: dict

    def find_student(self, query):
        query = str(query).strip()
        exact = [s for s in self.scores if s['student_id']==query]
        if exact: return exact
        def alias(v):
            v = re.sub(r'\s+', '', str(v))
            m = re.fullmatch(r'(?:학생)?(\d+)(?:번)?(?:학생)?', v)
            return str(int(m[1])) if m else v
        return [s for s in self.scores if alias(s['student_id'])==alias(query)]

    def question_context(self, number):
        if number not in self.key: raise ValueError('해당 문항이 없습니다.')
        record = self.questions['questions'][number-1]
        groups = self.questions['shared_contexts']
        spans = [s for g in record['shared_context_ids'] for s in groups[g]['spans']] + record['spans']
        return record, [self.images[s['image']] for s in spans]


def prepare_exam(student_bytes, answer_bytes, question_bytes):
    if not all([student_bytes, answer_bytes, question_bytes]):
        raise ValueError('세 종류의 파일을 모두 업로드해주세요.')
    if any(len(b)>30*1024*1024 for b in [student_bytes,answer_bytes,question_bytes]):
        raise ValueError('각 파일은 30MB 이하여야 합니다.')
    with PARSE_LOCK, tempfile.TemporaryDirectory(prefix='academy_exam_') as temp:
        root=Path(temp)
        paths=[root/'students.xlsx',root/'answers.pdf',root/'questions.pdf']
        for path,content in zip(paths,[student_bytes,answer_bytes,question_bytes]): path.write_bytes(content)
        key,km=parse_key(paths[1])
        students,sm=parse_students(paths[0],key)
        (root/'assets').mkdir()
        parsed=parse_questions(paths[2],key,root)
        if not km['form'] or km['form']!=parsed['form']:
            raise ValueError('문제지와 정답표의 홀수형/짝수형이 일치하지 않습니다.')
        images={str(p.relative_to(root)).replace('\\','/'):p.read_bytes() for p in (root/'assets').glob('*.png')}
    details,scores,stats=grade(students,key)
    return ExamData(fingerprint([student_bytes,answer_bytes,question_bytes]),key,students,parsed,images,
                    details,scores,stats,{'key':km,'students':sm})


def make_tools(exam):
    @tool
    def exam_summary() -> dict:
        """업로드된 시험 전체의 평균, 중앙값, 모집단 표준편차, 오답 순위를 조회한다."""
        return {'student_count':len(exam.students),'max_score':exam.metadata['key']['max_score'],
                'mean':exam.stats['mean'],'median':exam.stats['median'],
                'std':exam.stats['std_population'],'top_wrong':exam.stats['item_errors'][:3]}

    @tool
    def student_result(student_id: str) -> dict:
        """원본 학생명 또는 학생 번호로 점수와 오답 문항을 조회한다. 행 번호로 찾지 않는다."""
        matches=exam.find_student(student_id)
        if not matches: return {'error':'NOT_FOUND','student_id':student_id}
        if len(matches)>1:return {'error':'AMBIGUOUS','student_id':student_id}
        result=dict(matches[0])
        result['wrong_questions']=[r['question_id'] for r in exam.details if r['student_id']==result['student_id'] and not r['correct']]
        return result

    @tool
    def lowest_students() -> dict:
        """최저 점수와 동점자를 빠짐없이 조회한다."""
        return {'students':exam.stats['lowest_students']}

    @tool
    def item_result(question_id: int) -> dict:
        """문항의 정답, 배점, 오답 수와 선택지별 응답 수를 조회한다."""
        if question_id not in exam.key:return {'error':'NOT_FOUND'}
        row=next(r for r in exam.stats['item_errors'] if r['question_id']==question_id)
        rows=[r for r in exam.details if r['question_id']==question_id]
        return {**row,'answer':exam.key[question_id]['answer'],'points':exam.key[question_id]['points'],
                'distribution':{str(n):sum(r['response']==n for r in rows) for n in [None,1,2,3,4,5]}}
    return {t.name:t for t in [exam_summary,student_result,lowest_students,item_result]}


def excel_bytes(exam):
    """Overview, scores, item statistics and one detail sheet per student."""
    wb=Workbook(); overview=wb.active;overview.title='전체 성적 개요'
    def append(sheet, row):
        sheet.append(row)
        # Student labels remain literal text even if they start with '='.
        for cell in sheet[sheet.max_row]:
            if isinstance(cell.value,str):cell.data_type='s'
    for row in [('항목','값'),('학생 수',len(exam.students)),('문항 수',len(exam.key)),
                ('만점',exam.metadata['key']['max_score']),('평균',exam.stats['mean']),
                ('중앙값',exam.stats['median']),('표준편차 (모집단, ddof=0)',exam.stats['std_population']),
                ('오답 상위 3문항',', '.join(str(x['question_id']) for x in exam.stats['item_errors'][:3])),
                ('오답 동률 정렬','오답 수 내림차순, 문항번호 오름차순'),('무응답 처리','오답 / 0점')]: append(overview,row)
    for i, s in enumerate(exam.scores, 1):
        records = [r for r in exam.details if r["student_id"] == s["student_id"]]
       # Index avoids Excel's sheet-name collision and forbidden-character pitfalls.
        sheet=wb.create_sheet(f'학생_{i:03d}')
        append(sheet,['학생명',s['student_id']]);append(sheet,['총점',s['score']])
        append(sheet,['문항번호','제출 답','정답','배점','정답 여부','획득 점수'])
        for r in records:append(sheet,[r['question_id'],r['response'] if r['response'] is not None else '무응답',r['answer'],r['points'],'O' if r['correct'] else 'X',r['earned']])
        sheet.freeze_panes='B4';sheet.auto_filter.ref=f'A3:F{sheet.max_row}'
    for sheet in wb:
        header=3 if sheet.title.startswith('학생_') else 1
        for c in sheet[header]:c.font=Font(color='FFFFFF',bold=True);c.fill=PatternFill('solid',fgColor='176B61')
        if not sheet.freeze_panes:sheet.freeze_panes='B2'
        for col in sheet.columns:
            letter=col[0].column_letter
            sheet.column_dimensions[letter].width=min(65,max(15,max(len(str(c.value or ''))*1.4 for c in col)+2))
        for row in sheet:
            for cell in row:cell.alignment=Alignment(vertical='center')
    for n in [5,6,7]:overview.cell(n,2).number_format='0.00'
    stream=BytesIO();wb.save(stream);return stream.getvalue()
