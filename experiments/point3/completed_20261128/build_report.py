"""Compile the shared report template and create its Overleaf source ZIP."""
from pathlib import Path
import subprocess
import zipfile

REPORT = Path(__file__).resolve().parent/'report'


if __name__ == '__main__':
    for command in [['pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex'],
                    ['bibtex','main'],
                    ['pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex'],
                    ['pdflatex','-interaction=nonstopmode','-halt-on-error','main.tex']]:
        completed = subprocess.run(command,cwd=REPORT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        if completed.returncode:
            raise RuntimeError(completed.stdout[-8000:])
    log = (REPORT/'main.log').read_text(encoding='utf-8',errors='replace')
    for failure in ['undefined references','undefined citations','Overfull','Fatal error']:
        assert failure not in log, failure
    from pypdf import PdfReader
    assert all(':' not in page.extract_text() and ';' not in page.extract_text() for page in PdfReader(REPORT/'main.pdf').pages)
    archive = REPORT/'AdaptiveAttend_Overleaf.zip'
    excluded = {'.aux','.log','.out','.bbl','.blg','.fls','.fdb_latexmk','.zip','.pyc'}
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as stream:
        for path in sorted(REPORT.rglob('*')):
            if path.is_file() and path.suffix not in excluded and path.name != 'main.pdf' and '__pycache__' not in path.parts:
                stream.write(path,path.relative_to(REPORT).as_posix())
    print('Built',REPORT/'main.pdf')
    print('Packaged',archive)
