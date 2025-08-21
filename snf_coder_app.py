#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SNF Coding Assistant – Windows Desktop (Local/Domain Login, Keyboard-First)
- PyQt6 app, local-only processing, HIPAA-friendly
- Login Option 2:
    * DEFAULT: Local test login (username="12", password="34") — configurable in data/login_config.json
    * Domain login also available via LogonUserW; switch by editing login_config.json (mode: "domain")
- 3-attempt lockout, show/hide password, audit log (no passwords logged)
- Resizable UI (splitters), geometry persisted
- Light/Dark + “Slate Mint” theme
- Specialty picker + LARGE observation buttons (keyboard-friendly)
- Multi-format ingestion: PDF/DOCX/TXT/RTF/ODT/HTML
- Rule-based NLP (spaCy PhraseMatcher) + fallback keyword
- Local LLM Q&A (llama-cpp-python; auto-download Mistral-7B Q4 if needed)
- Medical chart–style PDF export + JSON backup
- Sample .txt auto-created if missing
- Robust QThread lifecycle (prevents “QThread destroyed…”)
"""

import os, sys, json, time
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

# -------- safe import --------
def try_import(name: str):
    try: return __import__(name)
    except Exception: return None

# -------- paths / settings --------
APP_NAME, ORG_NAME = "SNF_Coding_Assistant", "SNFApps"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR, DATA_DIR = os.path.join(BASE_DIR,"models"), os.path.join(BASE_DIR,"data")
os.makedirs(MODELS_DIR, exist_ok=True); os.makedirs(DATA_DIR, exist_ok=True)
SAMPLE_TXT_PATH = os.path.join(BASE_DIR, "sample_document.txt")
LOGIN_CFG_PATH = os.path.join(DATA_DIR, "login_config.json")

DEFAULT_MODEL_FILENAME = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
DEFAULT_MODEL_URL = ("https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/"
                     "mistral-7b-instruct-v0.2.Q4_K_M.gguf?download=true")
DEFAULT_MODEL_PATH = os.path.join(MODELS_DIR, DEFAULT_MODEL_FILENAME)

# -------- sample doc --------
SAMPLE_DEFAULT_TEXT = """Patient admitted for rehabilitation following acute ischemic stroke (CVA).
Diagnoses include dysphagia, Parkinson’s disease, and diabetes mellitus.
Diet order: mechanically altered diet with nectar thick liquids.
Nursing note: frequent coughing during meals, wet vocal quality.
Therapy note: patient requires moderate assistance with sit-to-stand transfers and bed mobility.
Orders: IV antibiotics for pneumonia (7-day course).
Skin assessment: unstageable pressure injury on sacrum; wound care consult placed.
Section K indicators: pocketing food, oral residue; patient reports pain when swallowing.
"""
def ensure_sample_txt():
    if not os.path.exists(SAMPLE_TXT_PATH):
        try:
            with open(SAMPLE_TXT_PATH,"w",encoding="utf-8") as f: f.write(SAMPLE_DEFAULT_TEXT)
        except Exception: pass
ensure_sample_txt()

# -------- login config helpers --------
DEFAULT_LOGIN_CFG = {
    "mode": "local",             # "local" or "domain"
    "local_user": "12",
    "local_password": "34"
}
def ensure_login_cfg():
    if not os.path.exists(LOGIN_CFG_PATH):
        try:
            with open(LOGIN_CFG_PATH, "w", encoding="utf-8") as f: json.dump(DEFAULT_LOGIN_CFG, f, indent=2)
        except Exception: pass
def load_login_cfg() -> Dict[str, str]:
    ensure_login_cfg()
    try:
        with open(LOGIN_CFG_PATH, "r", encoding="utf-8") as f: return json.load(f)
    except Exception:
        return DEFAULT_LOGIN_CFG.copy()

# -------- rules dictionaries --------
ICD10_TERMS = [
    {"canonical":"dysphagia","variants":["dysphagia","oropharyngeal dysphagia","swallowing disorder"],"category":"MDS_K0100"},
    {"canonical":"parkinsons_disease","variants":["parkinson's disease","parkinson disease","parkinson’s disease"],"category":"SLP_comorbidity"},
    {"canonical":"diabetes_mellitus","variants":["diabetes mellitus","type 2 diabetes","type ii diabetes","t2dm"],"category":"Diagnosis"},
    {"canonical":"pneumonia","variants":["pneumonia","pna"],"category":"Diagnosis"},
    {"canonical":"stroke","variants":["stroke","cva","ischemic stroke","acute ischemic stroke"],"category":"Diagnosis"},
]
MDS_SIGNALS = {
    "K0100":{"A":["loss of food","holding food","pocketing","oral residue"],
             "B":["coughing during meals","choking","wet voice","throat clearing","wet vocal quality"],
             "C":["odynophagia","globus","pain when swallowing"]},
    "Diet":{"mechanically_altered":["mechanically altered diet","pureed diet","minced & moist","minced and moist",
                                    "thickened liquids","nectar thick","honey thick"]},
    "GG":{"mobility":["sit-to-stand","sit to stand","bed mobility","transfer","walk 50 feet","moderate assistance"]}
}
NTA_CONDITIONS = [
    {"canonical":"hiv_aids","variants":["hiv","aids","antiretroviral therapy"]},
    {"canonical":"dialysis","variants":["hemodialysis","peritoneal dialysis","dialysis"]},
    {"canonical":"severe_wounds","variants":["stage 4 pressure ulcer","unstageable pressure injury","deep tissue injury"]},
    {"canonical":"iv_medications","variants":["iv antibiotics","intravenous antibiotics","parenteral nutrition","iv antifungal"]},
]

# -------- NLP --------
def build_spacy_matcher():
    spacy = try_import("spacy")
    if not spacy: return None,None,None
    try: nlp = spacy.blank("en")
    except Exception: nlp = spacy.blank("xx")
    from spacy.matcher import PhraseMatcher
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    catalog = []
    for sec,val in MDS_SIGNALS.items():
        if isinstance(val,dict):
            for sub,arr in val.items():
                for phrase in arr: catalog.append({"canonical":phrase,"variants":[phrase],"category":f"MDS_{sec}_{sub}"})
        else:
            for phrase in val: catalog.append({"canonical":phrase,"variants":[phrase],"category":f"MDS_{sec}"})
    catalog.extend(ICD10_TERMS)
    for item in NTA_CONDITIONS:
        i = dict(item); i["category"]="NTA_condition"; catalog.append(i)
    for it in catalog:
        docs = [nlp.make_doc(v) for v in it["variants"]]
        try: matcher.add(it["canonical"], docs)
        except Exception: pass
    meta={}
    for it in catalog:
        for v in it["variants"]: meta[v.lower()] = (it.get("canonical"), it.get("category"))
    return nlp, matcher, meta

# -------- extraction --------
def extract_text_from_path(path:str)->Tuple[str,str]:
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext==".pdf":
            pdfplumber = try_import("pdfplumber")
            if pdfplumber:
                with pdfplumber.open(path) as pdf:
                    pages=[p.extract_text() or "" for p in pdf.pages]
                return "\n".join(pages),"pdf"
            PyPDF2 = try_import("PyPDF2")
            if PyPDF2:
                text=[]; from PyPDF2 import PdfReader
                with open(path,"rb") as f:
                    r=PdfReader(f)
                    for p in r.pages:
                        try: text.append(p.extract_text() or "")
                        except Exception: text.append("")
                return "\n".join(text),"pdf"
        elif ext==".docx":
            docx = try_import("docx")
            if docx:
                from docx import Document
                d=Document(path); return "\n".join(p.text for p in d.paragraphs),"docx"
        elif ext==".rtf":
            srtf = try_import("striprtf")
            if srtf:
                from striprtf.striprtf import rtf_to_text
                with open(path,"r",encoding="utf-8",errors="ignore") as f: return rtf_to_text(f.read()),"rtf"
        elif ext==".odt":
            odf = try_import("odf")
            if odf:
                from odf.opendocument import load; from odf.text import P
                doc=load(path); txt=[]
                for p in doc.getElementsByType(P):
                    try: txt.append("".join(n.data for n in p.childNodes if hasattr(n,"data")))
                    except Exception: pass
                return "\n".join(txt),"odt"
        elif ext in [".html",".htm"]:
            bs4 = try_import("bs4")
            if bs4:
                from bs4 import BeautifulSoup
                with open(path,"r",encoding="utf-8",errors="ignore") as f: soup=BeautifulSoup(f.read(),"html.parser")
                return soup.get_text(separator="\n"),"html"
        elif ext in [".txt",""]:
            with open(path,"r",encoding="utf-8",errors="ignore") as f: return f.read(),"txt"
    except Exception: pass
    try:
        with open(path,"r",encoding="utf-8",errors="ignore") as f: return f.read(),"txt"
    except Exception: return "","unknown"

# -------- dataclasses / analysis --------
@dataclass
class Evidence:
    concept:str; canonical:str; variant:str; start:int; end:int; snippet:str; source:str; confidence:float=0.9
@dataclass
class Finding:
    category:str; concept:str; evidences:List[Evidence]; strength:str="medium"
@dataclass
class PDPMFlags:
    slp_indicators:List[str]; nta_candidates:List[str]; notes:List[str]
@dataclass
class Analysis:
    findings:List[Finding]; pdpm_flags:PDPMFlags; risk_scores:Dict[str,float]

def run_phrase_matcher(text:str, source:str)->List[Finding]:
    nlp,matcher,meta = build_spacy_matcher()
    if nlp and matcher and meta:
        doc=nlp.make_doc(text); matches=matcher(doc); grouped={}
        for mid,s,e in matches:
            var=doc[s:e].text; canonical,cat = meta.get(var.lower(),(var.lower(),"Unknown"))
            snip = doc[max(0,s-10):min(len(doc),e+10)].text
            ev = Evidence(canonical, canonical, var, s, e, snip, source, 0.9)
            grouped.setdefault((cat,canonical),[]).append(ev)
        return [Finding(category=k[0],concept=k[1],evidences=v,strength="medium") for k,v in grouped.items()]
    # fallback keyword
    grouped={}; low=text.lower()
    def add(variants, canonical, cat):
        for v in variants:
            s=0
            while True:
                i=low.find(v.lower(),s)
                if i==-1: break
                sn=text[max(0,i-50): i+len(v)+50]
                ev=Evidence(canonical,canonical,v,i,i+len(v),sn,source,0.6)
                grouped.setdefault((cat,canonical),[]).append(ev)
                s=i+len(v)
    for it in ICD10_TERMS: add(it["variants"], it["canonical"], it["category"])
    for it in NTA_CONDITIONS: add(it["variants"], it["canonical"], "NTA_condition")
    for sec,val in MDS_SIGNALS.items():
        if isinstance(val,dict):
            for sub,arr in val.items():
                for phrase in arr: add([phrase], phrase, f"MDS_{sec}_{sub}")
        else:
            for phrase in val: add([phrase], phrase, f"MDS_{sec}")
    return [Finding(category=k[0],concept=k[1],evidences=v,strength="low") for k,v in grouped.items()]

def run_rules(findings:List[Finding])->PDPMFlags:
    slp, nta, notes = [], [], []
    swallow = any(("MDS_K0100" in f.category or "dysphagia" in f.concept) for f in findings)
    mech_diet = any(("mechanically" in f.concept or "pureed" in f.concept or "minced" in f.concept or "thick" in f.concept)
                    and "MDS_Diet" in f.category for f in findings)
    slp_comorb = any(f.category=="SLP_comorbidity" for f in findings)
    if swallow: slp.append("Swallowing disorder evidence")
    if mech_diet: slp.append("Mechanically altered / thickened liquids referenced")
    if slp_comorb: slp.append("SLP-relevant comorbidity referenced")
    nta_hits = sorted({f.concept for f in findings if f.category=="NTA_condition"})
    if nta_hits: nta = nta_hits; notes.append("NTA candidates require verification (orders, labs, treatment dates).")
    return PDPMFlags(slp_indicators=slp, nta_candidates=nta, notes=notes)

def analyze_text(text:str, source:str)->Analysis:
    f = run_phrase_matcher(text, source)
    return Analysis(findings=f, pdpm_flags=run_rules(f), risk_scores={})

# -------- report (MD + PDF) --------
def build_markdown(analysis:Analysis, doc_name:str)->str:
    lines=[f"# Coding Assist Report — {doc_name}","","## PDPM Indicators (Preliminary)"]
    slp="; ".join(analysis.pdpm_flags.slp_indicators) if analysis.pdpm_flags.slp_indicators else "None found"
    nta=", ".join(analysis.pdpm_flags.nta_candidates) if analysis.pdpm_flags.nta_candidates else "None found"
    lines.append(f"- **SLP indicators:** {slp}")
    lines.append(f"- **NTA candidates:** {nta}")
    if analysis.pdpm_flags.notes: lines.append(f"_Notes:_ {' '.join(analysis.pdpm_flags.notes)}")
    lines.append(""); lines.append("## Evidence Table")
    for i,f in enumerate(analysis.findings,1):
        lines.append(f"### {i}. [{f.category}] {f.concept}")
        for e in f.evidences[:3]:
            lines.append(f'- Variant: "{e.variant}" | Pos: {e.start}–{e.end} | Snippet: {e.snippet.strip()}')
    return "\n".join(lines)

def narrative_summary(a:Analysis)->str:
    pts=[]
    if a.pdpm_flags.slp_indicators: pts.append(f"SLP indicators noted: {', '.join(a.pdpm_flags.slp_indicators)}.")
    if a.pdpm_flags.nta_candidates: pts.append(f"NTA candidates detected: {', '.join(a.pdpm_flags.nta_candidates)} (verify orders/dates).")
    if not pts: pts.append("No strong SLP indicators or NTA candidates detected by rules. Review documentation for completeness.")
    return " ".join(pts)

def recommendations_block(a:Analysis)->str:
    s=[]
    if a.pdpm_flags.slp_indicators: s.append("Confirm MDS Section K coding (K0100 A/B/C) with nursing/SLP and diet order dates.")
    if a.pdpm_flags.nta_candidates: s.append("Validate NTA items (e.g., IV meds, severe wounds) with orders, MAR, wound care notes.")
    s.append("Ensure diagnoses in problem list match current active conditions for I0020B and related fields.")
    s.append("Document functional status (Section GG) with precise assistance levels and distances.")
    return "\n".join("• "+x for x in s)

def red_flags_block(a:Analysis)->str:
    reds=[]
    has_dys = any("dysphagia" in f.concept for f in a.findings)
    has_pna = any(f.concept=="pneumonia" for f in a.findings)
    if has_dys and not a.pdpm_flags.slp_indicators:
        reds.append("Dysphagia mentioned but limited SLP indicators detected; verify swallow assessments and diet orders.")
    if has_pna: reds.append("Infection referenced (pneumonia); ensure correct start/stop dates for antibiotics.")
    if not reds: reds.append("No critical red flags detected by rules. Manual review recommended.")
    return "\n".join("• "+x for x in reds)

def export_pdf(analysis:Analysis, pdf_path:str, meta:Dict[str,str]):
    try:
        from reportlab.lib.pagesizes import LETTER
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    except Exception as e:
        raise RuntimeError("ReportLab required: pip install reportlab") from e
    doc = SimpleDocTemplate(pdf_path, pagesize=LETTER, rightMargin=36,leftMargin=36,topMargin=36,bottomMargin=36)
    styles=getSampleStyleSheet(); H1=styles["Heading1"]; H2=styles["Heading2"]; BODY=styles["BodyText"]
    BODY.spaceAfter=6; SMALL=ParagraphStyle("SMALL",parent=BODY,fontSize=9,leading=11,spaceAfter=4)
    flow=[Paragraph("SNF Coding Assist Report",H1)]
    cov=(f"<b>Facility:</b> {meta.get('facility','(not set)')}<br/>"
         f"<b>Patient ID:</b> {meta.get('patient_id','(not set)')}<br/>"
         f"<b>Document:</b> {meta.get('doc_name','(unknown)')}<br/>"
         f"<b>Date/Time:</b> {meta.get('datetime','')}<br/>")
    flow+= [Paragraph(cov,BODY), Spacer(1,0.25*inch), Paragraph("PDPM Indicators (Preliminary)",H2)]
    slp="; ".join(analysis.pdpm_flags.slp_indicators) if analysis.pdpm_flags.slp_indicators else "None found"
    nta=", ".join(analysis.pdpm_flags.nta_candidates) if analysis.pdpm_flags.nta_candidates else "None found"
    flow+= [Paragraph(f"<b>SLP indicators:</b> {slp}",BODY), Paragraph(f"<b>NTA candidates:</b> {nta}",BODY)]
    if analysis.pdpm_flags.notes: flow.append(Paragraph(f"<i>Notes:</i> {' '.join(analysis.pdpm_flags.notes)}",SMALL))
    flow.append(Spacer(1,0.2*inch))
    def table_section(title, rows):
        flow.append(Paragraph(title,H2))
        if not rows: flow.append(Paragraph("No items detected.",BODY)); return
        data=[["Category","Concept","Example Snippet"]]+rows[:25]
        t=Table(data,colWidths=[1.6*inch,2.2*inch,3.0*inch])
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
            ("GRID",(0,0),(-1,-1),0.5,colors.grey),("VALIGN",(0,0),(-1,-1),"TOP"),("FONTSIZE",(0,1),(-1,-1),9)
        ])); flow.append(t); flow.append(Spacer(1,0.2*inch))
    diag, slp_rows, nta_rows, other = [],[],[],[]
    for f in analysis.findings:
        sn=(f.evidences[0].snippet.strip().replace("\n"," ") if f.evidences else "")[:140] + \
           ("…" if (f.evidences and len(f.evidences[0].snippet)>140) else "")
        row=[f.category,f.concept,sn]
        if f.category=="Diagnosis": diag.append(row)
        elif f.category.startswith("MDS_K0100") or f.category.startswith("MDS_Diet"): slp_rows.append(row)
        elif f.category=="NTA_condition": nta_rows.append(row)
        else: other.append(row)
    table_section("Diagnoses",diag); table_section("Swallowing / Section K / SLP",slp_rows)
    table_section("NTA Candidates",nta_rows); table_section("Other / ICD-10-like",other)
    flow += [Paragraph("Narrative Summary",H2), Paragraph(narrative_summary(analysis).replace("\n","<br/>"),BODY),
             Spacer(1,0.1*inch), Paragraph("Recommendations",H2),
             Paragraph(recommendations_block(analysis).replace("\n","<br/>"),BODY),
             Paragraph("Red Flags / Alerts",H2), Paragraph(red_flags_block(analysis).replace("\n","<br/>"),BODY)]
    doc.build(flow)

# -------- simple retriever --------
class Retriever:
    def __init__(self):
        self._docs=[]
    def build(self, texts:List[str]): self._docs=texts[:]
    def query(self,q:str,top_k=4)->List[str]:
        if not self._docs: return []
        ql=q.lower(); scored=[]
        for i,d in enumerate(self._docs):
            hits=sum(1 for w in ql.split() if w in d.lower()); scored.append((hits,i))
        scored.sort(reverse=True); return [self._docs[i] for s,i in scored[:top_k] if s>0]

# -------- local LLM manager --------
class LLMManager:
    def __init__(self, model_path:str=DEFAULT_MODEL_PATH): self.model_path=model_path; self._llm=None; self._load_error=None
    def ensure_model(self, progress_cb=None)->bool:
        if os.path.exists(self.model_path) and os.path.getsize(self.model_path)>100_000_000: return True
        tmp=self.model_path+".part"
        try:
            req=try_import("requests"); 
            if not req: return False
            with req.get(DEFAULT_MODEL_URL, stream=True, timeout=60) as r:
                r.raise_for_status(); total=int(r.headers.get("content-length","0")); dl=0
                with open(tmp,"wb") as f:
                    for chunk in r.iter_content(chunk_size=1024*1024):
                        if not chunk: continue
                        f.write(chunk); dl+=len(chunk)
                        if progress_cb and total>0:
                            pct=int(dl*100/total); progress_cb(pct,f"Downloading model… {dl//(1024*1024)}MB / {total//(1024*1024)}MB")
            os.replace(tmp,self.model_path); return True
        except Exception:
            try:
                if os.path.exists(tmp): os.remove(tmp)
            except Exception: pass
            return False
    def load(self)->bool:
        if self._llm is not None: return True
        llama=try_import("llama_cpp")
        if not llama: self._load_error="Install llama-cpp-python"; return False
        try:
            from llama_cpp import Llama
            self._llm=Llama(model_path=self.model_path, n_ctx=4096, n_threads=os.cpu_count() or 4); return True
        except Exception as e:
            self._load_error=f"Failed to load model: {e}"; return False
    def chat(self, prompt:str, context:List[str], persona:str, max_tokens=512)->str:
        if not self._llm: return "(Model not loaded)"
        system=(f"You are a {persona} specializing in MDS/PDPM coding. Answer using ONLY the provided context. "
                "If context is insufficient, state what’s missing and next steps. Be concise.\n\n")
        ctx="\n\n--- CONTEXT START ---\n"+"\n\n".join(context[:6])+"\n--- CONTEXT END ---\n\n"
        final=system+ctx+"Question: "+prompt+"\nAnswer:"
        try:
            out=self._llm(final, max_tokens=max_tokens, temperature=0.2, stop=["\n\nQuestion:","###"])
            if isinstance(out,dict) and out.get("choices"): return out["choices"][0].get("text","").strip()
            return str(out)
        except Exception as e:
            return f"(LLM error) {e}"
# -------- Windows domain validation (only used when mode='domain') --------
def validate_windows_domain_credentials(user_or_domainuser: str, password: str) -> bool:
    try:
        import ctypes, ctypes.wintypes as wt
        advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
        LOGON32_LOGON_INTERACTIVE = 2
        LOGON32_PROVIDER_DEFAULT = 0
        phToken = wt.HANDLE()

        domain = None
        username = user_or_domainuser
        if "\\" in user_or_domainuser:
            domain, username = user_or_domainuser.split("\\", 1)
        # UPN (user@domain) -> domain=None

        res = advapi32.LogonUserW(
            wt.LPWSTR(username),
            wt.LPWSTR(domain) if domain else None,
            wt.LPWSTR(password),
            LOGON32_LOGON_INTERACTIVE,
            LOGON32_PROVIDER_DEFAULT,
            ctypes.byref(phToken)
        )
        if res == 0:
            return False
        # Close handle
        ctypes.windll.kernel32.CloseHandle(phToken)
        return True
    except Exception:
        # Secure default
        return False

# -------- PyQt6 UI --------
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QListWidget,
    QPushButton, QLabel, QFileDialog, QTextEdit, QPlainTextEdit, QLineEdit,
    QTabWidget, QProgressBar, QMessageBox, QStyleFactory, QSplitter, QCheckBox,
    QStatusBar, QComboBox, QDialog, QDialogButtonBox, QSizePolicy, QGridLayout
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QSettings
from PyQt6.QtGui import QPalette, QColor, QDragEnterEvent, QDropEvent, QKeySequence, QShortcut

# ---- Threads ----
class WorkerAnalyze(QThread):
    progress = pyqtSignal(int,str)
    finished_ok = pyqtSignal(object,str)   # (analysis, text, md), doc_name
    finished_err = pyqtSignal(str)
    def __init__(self,path:str): super().__init__(); self.path=path
    def run(self):
        try:
            if self.isInterruptionRequested(): return
            self.progress.emit(5,"Reading file…"); text, src = extract_text_from_path(self.path)
            if not text.strip(): raise RuntimeError("No text extracted; unsupported or empty file.")
            if self.isInterruptionRequested(): return
            self.progress.emit(35,"Running NLP rules…"); analysis = analyze_text(text, src)
            if self.isInterruptionRequested(): return
            self.progress.emit(80,"Building markdown…"); md=build_markdown(analysis, os.path.basename(self.path))
            if self.isInterruptionRequested(): return
            self.progress.emit(100,"Done"); self.finished_ok.emit((analysis,text,md), os.path.basename(self.path))
        except Exception as e:
            self.finished_err.emit(f"Analysis failed: {e}")

class WorkerDownloadModel(QThread):
    progress=pyqtSignal(int,str); finished_ok=pyqtSignal(); finished_err=pyqtSignal(str)
    def __init__(self,llm): super().__init__(); self.llm_mgr=llm
    def run(self):
        try:
            if self.isInterruptionRequested(): return
            ok=self.llm_mgr.ensure_model(progress_cb=lambda p,m: (None if self.isInterruptionRequested() else self.progress.emit(p,m)))
            if self.isInterruptionRequested(): return
            if not ok: raise RuntimeError("Auto-download failed. Download .gguf manually and select it.")
            self.progress.emit(100,"Model downloaded"); self.finished_ok.emit()
        except Exception as e:
            self.finished_err.emit(str(e))

class WorkerLLM(QThread):
    progress=pyqtSignal(int,str); finished_ok=pyqtSignal(str); finished_err=pyqtSignal(str)
    def __init__(self,llm,prompt,ctx,persona): super().__init__(); self.llm_mgr=llm; self.prompt=prompt; self.ctx=ctx; self.persona=persona
    def run(self):
        try:
            if self.isInterruptionRequested(): return
            self.progress.emit(5,"Loading model…")
            if not self.llm_mgr.load(): raise RuntimeError(self.llm_mgr._load_error or "Model load failed.")
            if self.isInterruptionRequested(): return
            self.progress.emit(50,"Generating answer…")
            ans=self.llm_mgr.chat(self.prompt, self.ctx, self.persona, max_tokens=512)
            if self.isInterruptionRequested(): return
            self.progress.emit(100,"Done"); self.finished_ok.emit(ans)
        except Exception as e:
            self.finished_err.emit(f"LLM error: {e}")

# ---- Login dialog (keyboard-first; local/domain configurable) ----
class LoginDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.cfg = load_login_cfg()
        self.setWindowTitle("Secure Login")
        self.setModal(True)
        self.setMinimumWidth(480)
        self._attempts = 0
        self._max_attempts = 3

        v = QVBoxLayout(self)
        mode_text = "LOCAL mode (test) — use configured username/password" if self.cfg.get("mode","local")=="local" \
            else "DOMAIN mode — enter DOMAIN\\username (or user@domain) and domain password"
        v.addWidget(QLabel(f"{mode_text}\nPress ENTER to submit."))

        self.user = QLineEdit()
        self.user.setPlaceholderText("Username" if self.cfg.get("mode")=="local" else "DOMAIN\\username  or  user@domain.tld")
        self.user.setClearButtonEnabled(True)
        if self.cfg.get("mode")=="local":
            self.user.setText(self.cfg.get("local_user",""))

        pw_row = QHBoxLayout()
        self.pw = QLineEdit()
        self.pw.setEchoMode(QLineEdit.EchoMode.Password)
        self.pw.setPlaceholderText("Password")
        if self.cfg.get("mode")=="local":
            self.pw.setText(self.cfg.get("local_password",""))
        self.chk_show = QCheckBox("Show")
        self.chk_show.stateChanged.connect(self.toggle_pw)
        pw_row.addWidget(self.pw, 1)
        pw_row.addWidget(self.chk_show, 0)

        v.addWidget(self.user)
        v.addLayout(pw_row)

        self.hint = QLabel(
            "Tip: To change login mode or credentials, edit data/login_config.json.\n"
            "LOCAL example: username '12', password '34' (defaults)."
        )
        self.hint.setStyleSheet("color: gray;")
        v.addWidget(self.hint)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.ok = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        self.ok.setText("Login")
        self.ok.setAutoDefault(True)
        self.ok.setDefault(True)
        v.addWidget(self.buttons)

        self.user.returnPressed.connect(self.focus_pw)
        self.pw.returnPressed.connect(self.try_login)
        self.buttons.accepted.connect(self.try_login)
        self.buttons.rejected.connect(self.reject)

        QShortcut(QKeySequence("Escape"), self, activated=self.reject)

        self.user.setFocus()

    def toggle_pw(self, state: int):
        self.pw.setEchoMode(QLineEdit.EchoMode.Normal if state == Qt.CheckState.Checked.value
                            else QLineEdit.EchoMode.Password)

    def focus_pw(self):
        self.pw.setFocus()
        self.pw.selectAll()

    def _audit(self, username: str, ok: bool):
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            with open(os.path.join(DATA_DIR, "auth.log"), "a", encoding="utf-8") as f:
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{ts}\tmode={self.cfg.get('mode')}\tuser={username}\tresult={'OK' if ok else 'FAIL'}\n")
        except Exception:
            pass

    def try_login(self):
        if self._attempts >= self._max_attempts:
            QMessageBox.critical(self, "Locked Out", "Too many failed attempts. Please close and reopen the app.")
            return

        u = self.user.text().strip()
        p = self.pw.text()
        if not u or not p:
            QMessageBox.warning(self, "Login", "Username and password required.")
            return

        mode = self.cfg.get("mode","local")
        if mode == "local":
            ok = (u == self.cfg.get("local_user","") and p == self.cfg.get("local_password",""))
        else:
            ok = validate_windows_domain_credentials(u, p)

        self._audit(u, ok)

        if not ok:
            self._attempts += 1
            remaining = self._max_attempts - self._attempts
            if remaining <= 0:
                QMessageBox.critical(self, "Locked Out", "Too many failed attempts. Please close and reopen the app.")
                self.reject()
                return
            msg = "Local login failed." if mode=="local" else "Domain authentication failed."
            QMessageBox.critical(self, "Login Failed", f"{msg}\nAttempts left: {remaining}")
            self.pw.clear()
            self.pw.setFocus()
            return

        self.accept()
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SNF Coding Assistant")
        self.resize(1280, 820)
        self.setAcceptDrops(True)

        self.current_path: Optional[str] = None
        self.current_text: str = ""
        self.current_source: str = ""
        self.current_doc_name: str = ""
        self.analysis: Optional[Analysis] = None
        self.markdown: str = ""
        self.worker: Optional[WorkerAnalyze] = None

        # Top bar
        top = QHBoxLayout()
        self.btn_open = QPushButton("Open…")
        self.btn_sample = QPushButton("Open Sample")
        self.btn_analyze = QPushButton("Analyze")
        self.btn_export = QPushButton("Export PDF…")
        top.addWidget(self.btn_open)
        top.addWidget(self.btn_sample)
        top.addWidget(self.btn_analyze)
        top.addWidget(self.btn_export)
        top.addStretch(1)
        top.addWidget(QLabel("Facility:"))
        self.input_facility = QLineEdit()
        self.input_facility.setPlaceholderText("Facility name")
        self.input_facility.setFixedWidth(200)
        top.addWidget(self.input_facility)
        top.addSpacing(8)
        top.addWidget(QLabel("Patient ID:"))
        self.input_patient = QLineEdit()
        self.input_patient.setPlaceholderText("ID")
        self.input_patient.setFixedWidth(140)
        top.addWidget(self.input_patient)
        top.addSpacing(16)
        top.addWidget(QLabel("Theme:"))
        self.theme = QComboBox()
        self.theme.addItems(["Light", "Dark", "Slate Mint"])
        top.addWidget(self.theme)

        # Body
        self.files = QListWidget()
        self.files.setSelectionMode(self.files.SelectionMode.SingleSelection)
        self.text_view = QPlainTextEdit()
        self.text_view.setReadOnly(True)
        self.report_view = QTextEdit()
        self.report_view.setReadOnly(True)

        right = QSplitter(Qt.Orientation.Vertical)
        right.addWidget(self.text_view)
        right.addWidget(self.report_view)
        right.setSizes([520, 300])

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.files)
        split.addWidget(right)
        split.setSizes([320, 960])
        self.splitter = split

        self.progress = QProgressBar()
        self.progress.setValue(0)

        central = QWidget()
        lay = QVBoxLayout(central)
        lay.addLayout(top)
        lay.addWidget(split)
        lay.addWidget(self.progress)
        self.setCentralWidget(central)

        self.status = QStatusBar()
        self.setStatusBar(self.status)

        # Signals
        self.btn_open.clicked.connect(self.open_file)
        self.btn_sample.clicked.connect(self.open_sample)
        self.btn_analyze.clicked.connect(self.run_analysis)
        self.btn_export.clicked.connect(self.do_export_pdf)
        self.files.itemSelectionChanged.connect(self.pick_from_list)
        self.theme.currentTextChanged.connect(self.apply_theme)

        # Restore settings
        self.settings = QSettings(ORG_NAME, APP_NAME)
        geom = self.settings.value("window/geometry", None)
        if geom:
            self.restoreGeometry(geom)
        state = self.settings.value("window/state", None)
        if state:
            self.restoreState(state)
        self.input_facility.setText(self.settings.value("meta/facility", ""))
        self.input_patient.setText(self.settings.value("meta/patient", ""))
        theme_name = self.settings.value("ui/theme", "Light")
        idx = self.theme.findText(theme_name)
        self.theme.setCurrentIndex(idx if idx >= 0 else 0)
        self.apply_theme(self.theme.currentText())

    # Window lifecycle
    def closeEvent(self, event):
        self.settings.setValue("window/geometry", self.saveGeometry())
        self.settings.setValue("window/state", self.saveState())
        self.settings.setValue("meta/facility", self.input_facility.text())
        self.settings.setValue("meta/patient", self.input_patient.text())
        self.settings.setValue("ui/theme", self.theme.currentText())
        super().closeEvent(event)

    # Drag & Drop
    def dragEnterEvent(self, e: QDragEnterEvent):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            e.ignore()

    def dropEvent(self, e: QDropEvent):
        urls = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if not urls:
            return
        for p in urls:
            if p and p not in [self.files.item(i).text() for i in range(self.files.count())]:
                self.files.addItem(p)
        self.load_path(urls[0])

    # Theme
    def apply_theme(self, which: str):
        if which == "Dark":
            self.setStyleSheet(
                """
                QWidget { background:#2d2d30; color:#e6e6e6; }
                QPlainTextEdit, QTextEdit, QListWidget { background:#1e1e1e; color:#e6e6e6; border:1px solid #444; }
                QPushButton { background:#3c3c3c; border:1px solid #555; padding:6px; }
                QPushButton:hover { background:#454545; }
                QProgressBar { background:#3c3c3c; border:1px solid #555; }
                QProgressBar::chunk { background:#007acc; }
                """
            )
        elif which == "Slate Mint":
            self.setStyleSheet(
                """
                QWidget { background:#f3f6f7; color:#1e2a2f; }
                QPlainTextEdit, QTextEdit, QListWidget { background:#ffffff; color:#1e2a2f; border:1px solid #c8d7da; }
                QPushButton { background:#e0f2ef; color:#0e5d4f; border:1px solid #a6d8cf; padding:6px; border-radius:4px; }
                QPushButton:hover { background:#c9ece6; }
                QProgressBar { background:#e9f2f4; border:1px solid #c8d7da; }
                QProgressBar::chunk { background:#2bb39a; }
                QStatusBar { background:#e9f2f4; border-top:1px solid #c8d7da; }
                """
            )
        else:
            self.setStyleSheet("")

    # File handling
    def open_sample(self):
        self.load_path(SAMPLE_TXT_PATH)

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Document",
            "",
            "All Supported (*.txt *.pdf *.docx *.rtf *.odt *.html *.htm);;"
            "Text (*.txt);;PDF (*.pdf);;Word (*.docx);;RTF (*.rtf);;ODT (*.odt);;HTML (*.html *.htm)",
        )
        if not path:
            return
        if path not in [self.files.item(i).text() for i in range(self.files.count())]:
            self.files.addItem(path)
        self.load_path(path)

    def pick_from_list(self):
        items = self.files.selectedItems()
        if not items:
            return
        self.load_path(items[0].text())

    def load_path(self, path: str):
        self.current_path = path
        self.current_doc_name = os.path.basename(path)
        self.progress.setValue(10)
        text, src = extract_text_from_path(path)
        self.current_text, self.current_source = text, src
        self.text_view.setPlainText(self.current_text)
        self.report_view.setMarkdown("Ready. Click Analyze.")
        self.progress.setValue(20)
        self.status.showMessage(f"Loaded {self.current_doc_name} ({src})", 3000)

    # Analyze (background)
    def run_analysis(self):
        if not self.current_path:
            QMessageBox.information(self, "Analyze", "Open a document first.")
            return
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "Analyze", "Analysis already running.")
            return
        self.progress.setValue(0)
        self.status.showMessage("Analyzing…")
        self.worker = WorkerAnalyze(self.current_path)
        self.worker.progress.connect(self.on_progress)
        self.worker.finished_ok.connect(self.on_analysis_ok)
        self.worker.finished_err.connect(self.on_analysis_err)
        self.worker.start()

    def on_progress(self, pct: int, msg: str):
        self.progress.setValue(pct)
        if msg:
            self.status.showMessage(msg)

    def on_analysis_ok(self, payload: object, doc_name: str):
        try:
            analysis, text, md = payload
        except Exception:
            QMessageBox.critical(self, "Analysis", "Unexpected analysis payload.")
            return
        self.analysis = analysis
        self.current_text = text
        self.current_doc_name = doc_name
        self.markdown = md
        self.text_view.setPlainText(self.current_text)
        self.report_view.setMarkdown(self.markdown)
        self.progress.setValue(100)
        self.status.showMessage("Analysis complete.", 3000)

    def on_analysis_err(self, err: str):
        self.progress.setValue(0)
        QMessageBox.critical(self, "Analysis Error", err)
        self.status.clearMessage()

    # Export
    def do_export_pdf(self):
        if not self.analysis:
            QMessageBox.information(self, "Export", "Nothing to export. Analyze first.")
            return
        out, _ = QFileDialog.getSaveFileName(self, "Save PDF", "report.pdf", "PDF (*.pdf)")
        if not out:
            return
        meta = {
            "facility": self.input_facility.text(),
            "patient_id": self.input_patient.text(),
            "doc_name": self.current_doc_name or "(unknown)",
            "datetime": time.strftime("%Y-%m-%d %H:%M"),
        }
        try:
            export_pdf(self.analysis, out, meta)
            QMessageBox.information(self, "Export", f"Saved: {out}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))


# -------------------------
# Main
# -------------------------
def main():
    app = QApplication(sys.argv)

    # Login first
    dlg = LoginDialog()
    if dlg.exec() != QDialog.DialogCode.Accepted:
        sys.exit(0)

    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
