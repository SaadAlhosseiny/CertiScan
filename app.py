import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from ELA import ELAEngine
from preprocessing import ImagePreprocessor
from noise import NoiseAnalyzer
from fft import FFTAnalyzer
from masking import create_adaptive_mask, morphological_refine, apply_red_overlay, combine_maps
from ai_detector import AIDetector
from dct_analyzer import DCTAnalyzer
from jpeg_ghost import JPEGGhostDetector
from copy_move import CopyMoveDetector
from API import chat
import time
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="CertiScan", page_icon="🔍", layout="wide", initial_sidebar_state="collapsed")

if "lang"         not in st.session_state: st.session_state.lang         = "ar"
if "page"         not in st.session_state: st.session_state.page         = "home"
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "groq_history" not in st.session_state: st.session_state.groq_history = []

L = {
    "ar": {
        "tagline": "الحقيقة لا تزور",
        "subtitle": "نظام جنائي رقمي متكامل لكشف التزوير في الوثائق والشهادات الرسمية",
        "nav_home": "الرئيسية", "nav_analyze": "التحليل", "nav_about": "عن النظام", "nav_chat": "المساعد",
        "start_btn": "ابدا التحليل الان",
        "upload_title": "ارفع وثيقتك هنا", "upload_sub": "JPG - JPEG - PNG",
        "upload_tip": "تاكد من وضوح الصورة وجودتها العالية",
        "file_name": "الاسم", "file_dim": "الابعاد", "file_size": "الحجم",
        "btn_analyze": "ابدا التحليل الجنائي", "btn_reset": "تحليل وثيقة اخرى",
        "p1": "تهيئة المنظومة...", "p2": "تحليل ELA...", "p3": "فحص البصمة الرقمية...",
        "p4": "تحليل الترددات...", "p5": "فحص DCT...", "p6": "كشف JPEG Ghost...",
        "p7": "كشف النسخ والصق...", "p8": "تحليل الذكاء الاصطناعي...", "p9": "اكتمل التحليل!",
        "tamper_label": "نسبة احتمال التعديل", "detail_title": "التحليل التفصيلي",
        "map_title": "خريطة المناطق المشبوهة",
        "map_legend": "احمر = عالي الشك  |  اصفر = متوسط  |  ازرق = طبيعي",
        "img_orig": "الاصلية", "img_ela": "ELA", "img_ghost": "JPEG Ghost",
        "img_copymove": "Copy-Move", "img_mask": "المناطق المشبوهة",
        "chat_title": "المحقق الذكي", "chat_sub": "اسالني اي شيء - مدعوم بالذكاء الاصطناعي",
        "chat_placeholder": "اكتب سؤالك هنا...", "chat_send": "ارسال", "chat_clear": "مسح المحادثة",
        "quick_q": "اسئلة سريعة:", "about_title": "عن نظام CertiScan", "lang_toggle": "English",
        "step1": "ارفع الوثيقة", "step2": "التحليل الجنائي", "step3": "النتيجة الفورية",
        "uploaded_img": "الوثيقة المرفوعة", "file_info": "معلومات الملف",
        "hero_stat1": "وثيقة محللة", "hero_stat2": "دقة الكشف", "hero_stat3": "تقنية جنائية",
        "how_works": "كيف يعمل النظام", "tech_used": "التقنيات المستخدمة",
        "verdict_clean": "سليمة", "verdict_sus": "مشبوهة", "verdict_forged": "مزورة",
        "desc_clean": "لم يتم اكتشاف اي علامات تزوير في هذه الوثيقة",
        "desc_sus": "تم اكتشاف بعض الانماط غير المعتادة - يُنصح بمراجعة يدوية",
        "desc_forged": "تم اكتشاف علامات واضحة على التلاعب والتعديل في الوثيقة",
        "scan_mode": "وضع الفحص الجنائي",
        "ai_powered": "مدعوم بالذكاء الاصطناعي",
    },
    "en": {
        "tagline": "Truth Cannot Be Forged",
        "subtitle": "Advanced digital forensic system for detecting document and certificate forgery",
        "nav_home": "Home", "nav_analyze": "Analysis", "nav_about": "About", "nav_chat": "Assistant",
        "start_btn": "Start Analysis Now",
        "upload_title": "Upload Your Document", "upload_sub": "JPG - JPEG - PNG",
        "upload_tip": "Ensure the image is clear and high quality",
        "file_name": "Name", "file_dim": "Dimensions", "file_size": "Size",
        "btn_analyze": "Start Forensic Analysis", "btn_reset": "Analyze Another",
        "p1": "Initializing system...", "p2": "ELA Analysis...", "p3": "Digital fingerprint...",
        "p4": "Frequency analysis...", "p5": "DCT Analysis...", "p6": "JPEG Ghost Detection...",
        "p7": "Copy-Move Detection...", "p8": "AI Analysis...", "p9": "Analysis Complete!",
        "tamper_label": "Tampering Probability", "detail_title": "Detailed Analysis",
        "map_title": "Suspicious Regions Map",
        "map_legend": "Red = High Risk  |  Yellow = Medium  |  Blue = Normal",
        "img_orig": "Original", "img_ela": "ELA", "img_ghost": "JPEG Ghost",
        "img_copymove": "Copy-Move", "img_mask": "Suspicious Regions",
        "chat_title": "AI Investigator", "chat_sub": "Ask me anything - Powered by AI",
        "chat_placeholder": "Type your question here...", "chat_send": "Send", "chat_clear": "Clear Chat",
        "quick_q": "Quick Questions:", "about_title": "About CertiScan", "lang_toggle": "عربي",
        "step1": "Upload Document", "step2": "Forensic Analysis", "step3": "Instant Result",
        "uploaded_img": "Uploaded Document", "file_info": "File Information",
        "hero_stat1": "Documents Analyzed", "hero_stat2": "Detection Accuracy", "hero_stat3": "Forensic Techniques",
        "how_works": "How It Works", "tech_used": "Technologies",
        "verdict_clean": "Authentic", "verdict_sus": "Suspicious", "verdict_forged": "Forged",
        "desc_clean": "No signs of forgery detected in this document",
        "desc_sus": "Unusual patterns detected - manual review recommended",
        "desc_forged": "Clear signs of tampering and manipulation detected",
        "scan_mode": "Forensic Scan Mode",
        "ai_powered": "AI Powered",
    }
}

t     = L[st.session_state.lang]
is_ar = st.session_state.lang == "ar"
dr    = "rtl" if is_ar else "ltr"
align = "right" if is_ar else "left"

quick_questions = {
    "ar": ["ما هو نظام CertiScan؟", "ما هي تقنية ELA؟", "كيف اقرا النتيجة؟", "ما هو JPEG Ghost؟", "ما هو Copy-Move؟", "كيف اكتشف التزوير؟"],
    "en": ["What is CertiScan?", "What is ELA technique?", "How do I read results?", "What is JPEG Ghost?", "What is Copy-Move?", "How to detect forgery?"]
}

def sc(s):
    if s < 12: return "#00ff88"
    elif s < 30: return "#ffaa00"
    return "#ff3366"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700;800;900&family=Rajdhani:wght@400;500;600;700&display=swap');

:root {{
    --cyan: #00f5ff;
    --green: #00ff88;
    --red: #ff3366;
    --yellow: #ffaa00;
    --purple: #b347ff;
    --bg: #010a0f;
    --bg2: #020f17;
    --card: rgba(0,245,255,0.03);
    --border: rgba(0,245,255,0.12);
}}

* {{ font-family: 'Exo 2', sans-serif !important; box-sizing: border-box; margin:0; padding:0; }}
.stApp {{ background: var(--bg) !important; color: #c8f0ff; overflow-x: hidden; }}

/* ── ANIMATED BACKGROUND ── */
.stApp::before {{
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 60% 40% at 15% 20%, rgba(0,245,255,0.06) 0%, transparent 55%),
        radial-gradient(ellipse 50% 60% at 85% 75%, rgba(179,71,255,0.05) 0%, transparent 55%),
        radial-gradient(ellipse 40% 30% at 50% 50%, rgba(0,255,136,0.03) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
    animation: bgPulse 8s ease-in-out infinite alternate;
}}
@keyframes bgPulse {{
    from {{ opacity: 0.7; }}
    to   {{ opacity: 1; }}
}}

/* ── MATRIX / FINGERPRINT CANVAS ── */
#matrix-canvas {{
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    pointer-events: none;
    z-index: 0;
    opacity: 0.07;
}}

/* ── SCAN LINES ── */
.stApp::after {{
    content: '';
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,245,255,0.015) 2px,
        rgba(0,245,255,0.015) 4px
    );
    pointer-events: none;
    z-index: 1;
    animation: scanMove 20s linear infinite;
}}
@keyframes scanMove {{
    from {{ background-position: 0 0; }}
    to   {{ background-position: 0 100vh; }}
}}

/* ── NAVBAR ── */
.cs-nav {{
    position: sticky; top: 0; z-index: 999;
    background: rgba(1,10,15,0.92);
    backdrop-filter: blur(20px);
    border-bottom: 1px solid var(--border);
    padding: 0 32px;
    display: flex; align-items: center; justify-content: space-between;
    height: 64px;
    direction: {dr};
}}
.cs-logo {{
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 22px;
    color: var(--cyan);
    letter-spacing: 3px;
    text-shadow: 0 0 20px rgba(0,245,255,0.6);
    animation: logoPulse 3s ease-in-out infinite;
}}
@keyframes logoPulse {{
    0%,100% {{ text-shadow: 0 0 20px rgba(0,245,255,0.6); }}
    50%      {{ text-shadow: 0 0 40px rgba(0,245,255,1), 0 0 80px rgba(0,245,255,0.4); }}
}}
.cs-status {{
    display: flex; align-items: center; gap: 8px;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 11px; color: var(--green);
    letter-spacing: 2px;
}}
.cs-dot {{
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--green);
    animation: dotBlink 1.5s ease-in-out infinite;
    box-shadow: 0 0 8px var(--green);
}}
@keyframes dotBlink {{ 0%,100%{{opacity:1}} 50%{{opacity:0.3}} }}

/* ── HERO ── */
.hero-wrap {{
    text-align: center;
    padding: 80px 20px 50px;
    position: relative;
    direction: {dr};
}}
.hero-eyebrow {{
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 11px; letter-spacing: 6px;
    color: var(--cyan); margin-bottom: 20px;
    opacity: 0; animation: fadeUp 0.8s 0.2s forwards;
}}
.hero-title {{
    font-family: 'Share Tech Mono', monospace !important;
    font-size: clamp(56px, 10vw, 120px);
    font-weight: 900; line-height: 0.9;
    background: linear-gradient(135deg, #ffffff 0%, var(--cyan) 40%, var(--purple) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    filter: drop-shadow(0 0 30px rgba(0,245,255,0.3));
    opacity: 0; animation: titleReveal 1s 0.4s forwards;
    letter-spacing: 4px;
}}
@keyframes titleReveal {{
    from {{ opacity:0; transform: translateY(-30px) scale(0.95); }}
    to   {{ opacity:1; transform: translateY(0) scale(1); }}
}}
.hero-tagline {{
    font-size: clamp(16px, 2.5vw, 24px);
    color: var(--cyan); font-weight: 300;
    letter-spacing: 2px; margin: 16px 0 12px;
    opacity: 0; animation: fadeUp 0.8s 0.6s forwards;
    font-family: 'Rajdhani', sans-serif !important;
}}
.hero-sub {{
    font-size: 15px; color: #4a8fa8;
    max-width: 560px; margin: 0 auto 40px;
    line-height: 1.8;
    opacity: 0; animation: fadeUp 0.8s 0.8s forwards;
}}
@keyframes fadeUp {{
    from {{ opacity:0; transform: translateY(20px); }}
    to   {{ opacity:1; transform: translateY(0); }}
}}

/* ── STATS ── */
.stats-row {{
    display: flex; justify-content: center; gap: 60px;
    margin: 30px 0 40px;
    opacity: 0; animation: fadeUp 0.8s 1s forwards;
    flex-wrap: wrap;
}}
.stat-item {{ text-align: center; }}
.stat-num {{
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 42px; font-weight: 900;
    background: linear-gradient(135deg, var(--cyan), var(--purple));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    filter: drop-shadow(0 0 10px rgba(0,245,255,0.4));
}}
.stat-label {{ font-size: 11px; color: #2a6a80; letter-spacing: 2px; margin-top: 4px; text-transform: uppercase; }}

/* ── STEP CARDS ── */
.step-grid {{ display: grid; grid-template-columns: repeat(3,1fr); gap: 16px; direction: {dr}; }}
.step-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px; padding: 28px 20px;
    text-align: center; position: relative; overflow: hidden;
    transition: all 0.4s;
    cursor: default;
}}
.step-card::before {{
    content: ''; position: absolute;
    top: -2px; left: -2px; right: -2px; bottom: -2px;
    background: linear-gradient(135deg, var(--cyan), var(--purple), var(--cyan));
    border-radius: 18px; z-index: -1;
    opacity: 0; transition: opacity 0.4s;
    background-size: 200% 200%;
    animation: borderAnim 3s linear infinite;
}}
@keyframes borderAnim {{
    0%   {{ background-position: 0% 50%; }}
    50%  {{ background-position: 100% 50%; }}
    100% {{ background-position: 0% 50%; }}
}}
.step-card:hover::before {{ opacity: 1; }}
.step-card:hover {{ transform: translateY(-6px); background: rgba(0,245,255,0.06); }}
.step-num {{
    width: 44px; height: 44px;
    border: 2px solid var(--cyan);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 20px; color: var(--cyan);
    margin: 0 auto 14px;
    box-shadow: 0 0 15px rgba(0,245,255,0.3);
}}
.step-icon {{ font-size: 36px; margin-bottom: 12px; }}
.step-label {{ font-size: 14px; font-weight: 600; color: #8ab8c8; letter-spacing: 1px; }}

/* ── TECH CARDS ── */
.tech-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px; padding: 24px 16px;
    text-align: center; transition: all 0.3s;
    position: relative; overflow: hidden;
}}
.tech-card::after {{
    content: ''; position: absolute;
    bottom: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--cyan), var(--purple));
    transform: scaleX(0); transition: transform 0.3s;
}}
.tech-card:hover::after {{ transform: scaleX(1); }}
.tech-card:hover {{ border-color: rgba(0,245,255,0.3); transform: translateY(-4px); }}

/* ── UPLOAD ZONE ── */
.upload-zone {{
    border: 2px dashed rgba(0,245,255,0.25);
    border-radius: 20px; padding: 60px 30px;
    text-align: center;
    background: rgba(0,245,255,0.02);
    transition: all 0.4s; direction: {dr};
    position: relative; overflow: hidden;
}}
.upload-zone::before {{
    content: ''; position: absolute; inset: 0;
    background: radial-gradient(ellipse at center, rgba(0,245,255,0.04) 0%, transparent 70%);
    opacity: 0; transition: opacity 0.4s;
}}
.upload-zone:hover::before {{ opacity: 1; }}
.upload-zone:hover {{ border-color: rgba(0,245,255,0.5); }}

/* ── RESULT CARD ── */
.result-card {{
    border-radius: 24px; padding: 60px 40px;
    text-align: center; margin: 28px 0;
    position: relative; overflow: hidden;
    animation: resultReveal 0.8s cubic-bezier(0.34,1.56,0.64,1);
    direction: {dr};
}}
@keyframes resultReveal {{
    from {{ opacity:0; transform: scale(0.8) translateY(30px); }}
    to   {{ opacity:1; transform: scale(1) translateY(0); }}
}}
.result-clean  {{
    background: linear-gradient(135deg, rgba(0,255,136,0.1), rgba(0,255,136,0.03));
    border: 2px solid rgba(0,255,136,0.5);
    box-shadow: 0 0 80px rgba(0,255,136,0.15), inset 0 0 60px rgba(0,255,136,0.05);
}}
.result-sus    {{
    background: linear-gradient(135deg, rgba(255,170,0,0.1), rgba(255,170,0,0.03));
    border: 2px solid rgba(255,170,0,0.5);
    box-shadow: 0 0 80px rgba(255,170,0,0.15), inset 0 0 60px rgba(255,170,0,0.05);
}}
.result-forged {{
    background: linear-gradient(135deg, rgba(255,51,102,0.1), rgba(255,51,102,0.03));
    border: 2px solid rgba(255,51,102,0.5);
    box-shadow: 0 0 80px rgba(255,51,102,0.2), inset 0 0 60px rgba(255,51,102,0.05);
}}
.big-pct {{
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 120px; font-weight: 900; line-height: 1;
    filter: drop-shadow(0 0 30px currentColor);
    animation: pctReveal 1s 0.3s backwards;
}}
@keyframes pctReveal {{
    from {{ opacity:0; transform: scale(0.5); }}
    to   {{ opacity:1; transform: scale(1); }}
}}
.verdict-text {{
    font-size: 42px; font-weight: 800;
    margin-top: 12px; letter-spacing: 3px;
    font-family: 'Rajdhani', sans-serif !important;
    animation: fadeUp 0.6s 0.5s backwards;
}}
.verdict-sub {{ font-size: 15px; color: #4a8fa8; margin-top: 10px; animation: fadeUp 0.6s 0.7s backwards; }}

/* ── METRIC CARDS ── */
.metric-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px; padding: 24px 14px;
    text-align: center; transition: all 0.3s;
    position: relative;
}}
.metric-card:hover {{
    border-color: rgba(0,245,255,0.3);
    transform: translateY(-5px);
    box-shadow: 0 12px 40px rgba(0,245,255,0.1);
}}
.metric-num {{
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 44px; font-weight: 900;
    filter: drop-shadow(0 0 8px currentColor);
}}
.metric-name {{ font-size: 13px; font-weight: 700; color: #4a8fa8; margin-top: 8px; letter-spacing: 2px; text-transform: uppercase; }}
.metric-desc {{ font-size: 11px; color: #2a5060; margin-top: 4px; }}

/* ── ABOUT CARDS ── */
.about-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px; padding: 28px;
    margin-bottom: 16px; direction: {dr};
    transition: all 0.3s;
    position: relative; overflow: hidden;
}}
.about-card::before {{
    content: ''; position: absolute;
    left: 0; top: 0; bottom: 0; width: 3px;
    background: linear-gradient(180deg, var(--cyan), var(--purple));
    opacity: 0; transition: opacity 0.3s;
}}
.about-card:hover::before {{ opacity: 1; }}
.about-card:hover {{ border-color: rgba(0,245,255,0.2); padding-left: 32px; }}

/* ── CHAT ── */
.msg-user {{
    background: linear-gradient(135deg, rgba(0,245,255,0.1), rgba(179,71,255,0.1));
    border: 1px solid rgba(0,245,255,0.2);
    border-radius: 16px 16px {'4px 16px' if is_ar else '16px 4px'};
    padding: 12px 16px; font-size: 14px;
    margin: 6px 0; text-align: {align};
    font-family: 'Rajdhani', sans-serif !important;
}}
.msg-bot {{
    background: rgba(0,245,255,0.03);
    border: 1px solid rgba(0,245,255,0.1);
    border-radius: 16px 16px {'16px 4px' if is_ar else '4px 16px'};
    padding: 14px 18px; font-size: 14px;
    line-height: 1.8; margin: 6px 0; text-align: {align};
}}
.bot-label {{
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 10px; color: var(--cyan);
    margin-bottom: 6px; letter-spacing: 2px;
}}

/* ── FILE INFO ── */
.file-info-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 16px; padding: 24px;
    direction: {dr};
    font-family: 'Share Tech Mono', monospace !important;
}}
.fi-label {{ font-size: 10px; color: #2a6a80; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 4px; }}
.fi-value {{ font-size: 14px; color: var(--cyan); margin-bottom: 16px; }}

/* ── PROGRESS ── */
.stProgress > div > div {{
    background: linear-gradient(90deg, var(--cyan), var(--purple)) !important;
    border-radius: 4px !important;
    box-shadow: 0 0 10px rgba(0,245,255,0.5) !important;
    animation: progressGlow 1s ease-in-out infinite alternate !important;
}}
@keyframes progressGlow {{
    from {{ box-shadow: 0 0 10px rgba(0,245,255,0.5); }}
    to   {{ box-shadow: 0 0 20px rgba(0,245,255,0.9), 0 0 40px rgba(179,71,255,0.4); }}
}}

/* ── BUTTONS ── */
.stButton > button {{
    background: transparent !important;
    color: var(--cyan) !important;
    font-size: 15px !important;
    font-weight: 700 !important;
    padding: 14px 28px !important;
    border-radius: 10px !important;
    border: 1px solid rgba(0,245,255,0.4) !important;
    width: 100% !important;
    transition: all 0.3s !important;
    font-family: 'Rajdhani', sans-serif !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    position: relative !important;
    overflow: hidden !important;
}}
.stButton > button::before {{
    content: '' !important;
    position: absolute !important;
    inset: 0 !important;
    background: linear-gradient(135deg, rgba(0,245,255,0.1), rgba(179,71,255,0.1)) !important;
    opacity: 0 !important;
    transition: opacity 0.3s !important;
}}
.stButton > button:hover {{
    border-color: var(--cyan) !important;
    box-shadow: 0 0 20px rgba(0,245,255,0.3), inset 0 0 20px rgba(0,245,255,0.05) !important;
    transform: translateY(-2px) !important;
}}

/* ── SECTION DIVIDER ── */
.cs-divider {{
    display: flex; align-items: center; gap: 16px;
    margin: 32px 0 20px; direction: {dr};
}}
.cs-divider-line {{ flex: 1; height: 1px; background: linear-gradient(90deg, transparent, var(--border), transparent); }}
.cs-divider-text {{
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 10px; color: #2a6a80; letter-spacing: 3px; text-transform: uppercase;
    white-space: nowrap;
}}

hr {{ border-color: rgba(0,245,255,0.08) !important; }}
[data-testid="stFileUploader"] {{ background: transparent !important; }}

/* ── CORNER DECORATIONS ── */
.corner-tl, .corner-tr, .corner-bl, .corner-br {{
    position: absolute; width: 20px; height: 20px;
    border-color: var(--cyan); border-style: solid; opacity: 0.5;
}}
.corner-tl {{ top: 12px; left: 12px; border-width: 2px 0 0 2px; border-radius: 4px 0 0 0; }}
.corner-tr {{ top: 12px; right: 12px; border-width: 2px 2px 0 0; border-radius: 0 4px 0 0; }}
.corner-bl {{ bottom: 12px; left: 12px; border-width: 0 0 2px 2px; border-radius: 0 0 0 4px; }}
.corner-br {{ bottom: 12px; right: 12px; border-width: 0 2px 2px 0; border-radius: 0 0 4px 0; }}
</style>

<!-- Matrix Canvas -->
<canvas id="matrix-canvas"></canvas>
<script>
(function() {{
    const canvas = document.getElementById('matrix-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const chars = '01アイウエオカキクケコ∑∆∏√∞≠≈∫∂∇⊕⊗⊙◈◉◎▲▼◆●■';
    const fontSize = 13;
    const cols = Math.floor(canvas.width / fontSize);
    const drops = Array(cols).fill(1);

    function draw() {{
        ctx.fillStyle = 'rgba(1,10,15,0.05)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = 'rgba(0,245,255,0.5)';
        ctx.font = fontSize + 'px Share Tech Mono, monospace';
        for (let i = 0; i < drops.length; i++) {{
            const text = chars[Math.floor(Math.random() * chars.length)];
            ctx.fillText(text, i * fontSize, drops[i] * fontSize);
            if (drops[i] * fontSize > canvas.height && Math.random() > 0.975) drops[i] = 0;
            drops[i]++;
        }}
    }}
    setInterval(draw, 50);
    window.addEventListener('resize', () => {{
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }});
}})();
</script>
""", unsafe_allow_html=True)

# ── NAVBAR ──
st.markdown(f"""
<div class="cs-nav">
    <div class="cs-logo">CERTISCAN</div>
    <div class="cs-status">
        <div class="cs-dot"></div>
        {'نظام نشط' if is_ar else 'SYSTEM ACTIVE'}
    </div>
</div>
""", unsafe_allow_html=True)

# ── NAV BUTTONS ──
n1,n2,n3,n4,n5 = st.columns([1,1,1,1,1])
for col, pg, label in [
    (n1,"home",t["nav_home"]),
    (n2,"analyze",t["nav_analyze"]),
    (n3,"about",t["nav_about"]),
    (n4,"chat",t["nav_chat"]),
    (n5,"lang",""),
]:
    with col:
        if pg == "lang":
            if st.button(t["lang_toggle"], key="lang_btn"):
                st.session_state.lang = "en" if st.session_state.lang=="ar" else "ar"
                st.rerun()
        else:
            if st.button(label, key=f"nav_{pg}"):
                st.session_state.page = pg
                st.rerun()

# ════════════════════════════════════════════
#  HOME
# ════════════════════════════════════════════
if st.session_state.page == "home":
    st.markdown(f"""
    <div class="hero-wrap">
        <div class="hero-eyebrow">[ FORENSIC DOCUMENT ANALYSIS SYSTEM v3.0 ]</div>
        <div class="hero-title">CERTISCAN</div>
        <div class="hero-tagline">{t["tagline"]}</div>
        <div class="hero-sub">{t["subtitle"]}</div>
        <div class="stats-row">
            <div class="stat-item"><div class="stat-num">10K+</div><div class="stat-label">{t["hero_stat1"]}</div></div>
            <div class="stat-item"><div class="stat-num">97%</div><div class="stat-label">{t["hero_stat2"]}</div></div>
            <div class="stat-item"><div class="stat-num">07</div><div class="stat-label">{t["hero_stat3"]}</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    _, cc, _ = st.columns([1,2,1])
    with cc:
        if st.button(f"⟩  {t['start_btn']}", key="hero_start"):
            st.session_state.page = "analyze"
            st.rerun()

    st.markdown(f"""
    <div class="cs-divider">
        <div class="cs-divider-line"></div>
        <div class="cs-divider-text">{t["how_works"]}</div>
        <div class="cs-divider-line"></div>
    </div>
    <div class="step-grid">
        <div class="step-card">
            <div class="step-num">01</div>
            <div class="step-icon">📤</div>
            <div class="step-label">{t["step1"]}</div>
        </div>
        <div class="step-card">
            <div class="step-num">02</div>
            <div class="step-icon">🔬</div>
            <div class="step-label">{t["step2"]}</div>
        </div>
        <div class="step-card">
            <div class="step-num">03</div>
            <div class="step-icon">📊</div>
            <div class="step-label">{t["step3"]}</div>
        </div>
    </div>
    <div class="cs-divider" style="margin-top:40px">
        <div class="cs-divider-line"></div>
        <div class="cs-divider-text">{t["tech_used"]}</div>
        <div class="cs-divider-line"></div>
    </div>
    """, unsafe_allow_html=True)

    tc1,tc2,tc3,tc4 = st.columns(4)
    for col,ic,name,desc,color in [
        (tc1,"🔬","ELA","Error Level Analysis","#00f5ff"),
        (tc2,"🌊","FFT+DCT","Frequency Analysis","#b347ff"),
        (tc3,"👻","JPEG Ghost","Compression Analysis","#00ff88"),
        (tc4,"🔁","Copy-Move","Duplication Detection","#ffaa00"),
    ]:
        with col:
            st.markdown(f'<div class="tech-card"><div style="font-size:28px;margin-bottom:10px">{ic}</div><div style="font-size:16px;font-weight:800;color:{color};margin-bottom:6px;letter-spacing:1px">{name}</div><div style="font-size:11px;color:#2a6a80;letter-spacing:1px">{desc}</div></div>', unsafe_allow_html=True)

# ════════════════════════════════════════════
#  ANALYZE
# ════════════════════════════════════════════
elif st.session_state.page == "analyze":
    st.markdown(f"""
    <div style="direction:{dr};margin:24px 0 20px;position:relative">
        <div style="font-family:'Share Tech Mono',monospace;font-size:10px;color:#2a6a80;letter-spacing:4px;margin-bottom:8px">
            [ {t["scan_mode"]} ]
        </div>
        <div style="font-size:28px;font-weight:800;color:#c8f0ff;letter-spacing:1px">
            🔬 {'تحليل الوثيقة' if is_ar else 'Document Analysis'}
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed")

    if not uploaded_file:
        st.markdown(f"""
        <div class="upload-zone">
            <div style="font-size:64px;margin-bottom:20px;filter:drop-shadow(0 0 20px rgba(0,245,255,0.4))">📂</div>
            <div style="font-size:22px;font-weight:700;color:#c8f0ff;margin-bottom:8px">{t["upload_title"]}</div>
            <div style="font-family:'Share Tech Mono',monospace;font-size:12px;color:#2a6a80;letter-spacing:3px">{t["upload_sub"]}</div>
            <div style="margin-top:20px;font-size:12px;color:rgba(0,245,255,0.5);letter-spacing:1px">⚠ {t["upload_tip"]}</div>
        </div>
        """, unsafe_allow_html=True)

    if uploaded_file:
        Path("temp").mkdir(exist_ok=True)
        temp_input = Path("temp") / uploaded_file.name
        temp_input.write_bytes(uploaded_file.getvalue())
        img_bgr = cv2.imread(str(temp_input))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_bgr.shape[:2]

        ci, cf = st.columns([2,1])
        with ci:
            st.image(img_rgb, caption=t["uploaded_img"], use_column_width=True)
        with cf:
            st.markdown(f"""
            <div class="file-info-card">
                <div style="font-size:11px;color:rgba(0,245,255,0.5);letter-spacing:3px;margin-bottom:16px">> {t["file_info"]}</div>
                <div class="fi-label">{t["file_name"]}</div>
                <div class="fi-value" style="font-size:12px;word-break:break-all">{uploaded_file.name}</div>
                <div class="fi-label">{t["file_dim"]}</div>
                <div class="fi-value">{w} x {h} px</div>
                <div class="fi-label">{t["file_size"]}</div>
                <div class="fi-value">{uploaded_file.size//1024} KB</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button(f"⟩⟩  {t['btn_analyze']}", use_container_width=True):
            prog   = st.progress(0)
            status = st.empty()

            preprocessor   = ImagePreprocessor(target_size=(512,512))
            ela_engine     = ELAEngine()
            noise_analyzer = NoiseAnalyzer()
            fft_analyzer   = FFTAnalyzer()
            dct_analyzer   = DCTAnalyzer()
            ghost_detector = JPEGGhostDetector()
            cm_detector    = CopyMoveDetector()
            ai_detector    = AIDetector()

            steps = [(t["p1"],8),(t["p2"],20),(t["p3"],34),(t["p4"],48),(t["p5"],60),(t["p6"],72),(t["p7"],84),(t["p8"],94)]
            for msg, pct in steps:
                status.markdown(f"""
                <div style="text-align:center;padding:12px">
                    <span style="font-family:'Share Tech Mono',monospace;font-size:12px;color:#00f5ff;letter-spacing:2px">
                        ▸ {msg}
                    </span>
                </div>
                """, unsafe_allow_html=True)
                prog.progress(pct)
                time.sleep(0.2)

            img_clean = preprocessor.preprocess(str(temp_input))
            img_gray  = cv2.cvtColor(img_clean, cv2.COLOR_RGB2GRAY)

            ela_display, _, diff_raw = ela_engine.calculate_ela(img_clean)
            ela_gray  = cv2.cvtColor(diff_raw, cv2.COLOR_RGB2GRAY) if len(diff_raw.shape)==3 else diff_raw
            ela_score = min(float(np.percentile(ela_gray, 95)) / 255 * 100, 100)

            noise_map, _ = noise_analyzer.analyze_noise(img_gray)
            noise_score  = noise_analyzer.get_noise_score(noise_map)

            fft_map   = fft_analyzer.analyze_fft(img_gray)
            fft_score = fft_analyzer.get_fft_score(fft_map)

            dct_map   = dct_analyzer.analyze(img_gray)
            dct_score = dct_analyzer.score(dct_map)

            ghost_map   = ghost_detector.analyze(img_bgr)
            ghost_score = ghost_detector.score(ghost_map)

            cm_map, cm_score = cm_detector.detect(img_gray)
            ai_score         = ai_detector.score(img_gray)

            combined   = combine_maps(diff_raw, fft_map, noise_map, weights=[0.3, 0.5, 0.2])
            mask_raw   = create_adaptive_mask(combined, method="triangle")
            mask_ref   = morphological_refine(mask_raw, erode_ksize=5, erode_iterations=2)
            mask_score = min(float(np.sum(mask_ref > 0) / mask_ref.size * 100) * 5, 100)

            final_score = round(min(
                0.25*ela_score + 0.20*fft_score + 0.15*noise_score +
                0.10*dct_score + 0.15*ghost_score + 0.10*cm_score + 0.05*mask_score
            , 100), 1)

            prog.progress(100)
            status.markdown(f"""
            <div style="text-align:center;padding:12px">
                <span style="font-family:'Share Tech Mono',monospace;font-size:12px;color:#00ff88;letter-spacing:2px">
                    ✓ {t["p9"]}
                </span>
            </div>
            """, unsafe_allow_html=True)
            time.sleep(0.5)
            prog.empty(); status.empty()

            if final_score < 12:
                css, icon, verdict, color = "result-clean",  "✓", t["verdict_clean"],  "#00ff88"
                desc = t["desc_clean"]
            elif final_score < 30:
                css, icon, verdict, color = "result-sus",    "⚠", t["verdict_sus"],    "#ffaa00"
                desc = t["desc_sus"]
            else:
                css, icon, verdict, color = "result-forged", "✗", t["verdict_forged"], "#ff3366"
                desc = t["desc_forged"]

            st.markdown(f"""
            <div class="result-card {css}">
                <div class="corner-tl"></div><div class="corner-tr"></div>
                <div class="corner-bl"></div><div class="corner-br"></div>
                <div class="big-pct" style="color:{color}">{final_score}%</div>
                <div class="verdict-text" style="color:{color}">{icon} {verdict}</div>
                <div class="verdict-sub">{desc}</div>
                <div style="margin-top:16px;font-family:'Share Tech Mono',monospace;font-size:11px;color:#2a6a80;letter-spacing:2px">
                    {t["tamper_label"]}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="cs-divider">
                <div class="cs-divider-line"></div>
                <div class="cs-divider-text">{t["detail_title"]}</div>
                <div class="cs-divider-line"></div>
            </div>
            """, unsafe_allow_html=True)

            m1,m2,m3,m4 = st.columns(4)
            for col,ic,nm,desc_t,s in [
                (m1,"🔬","ELA","Error Level", round(ela_score,1)),
                (m2,"📡","Noise","Fingerprint",  round(noise_score,1)),
                (m3,"🌊","FFT","Frequency",     round(fft_score,1)),
                (m4,"📐","DCT","Block Analysis", round(dct_score,1)),
            ]:
                with col:
                    st.markdown(f'<div class="metric-card"><div style="font-size:22px">{ic}</div><div class="metric-num" style="color:{sc(s)}">{s}%</div><div class="metric-name">{nm}</div><div class="metric-desc">{desc_t}</div></div>', unsafe_allow_html=True)
                    st.progress(min(int(s),100))

            m5,m6,m7,m8 = st.columns(4)
            for col,ic,nm,desc_t,s in [
                (m5,"👻","Ghost","JPEG Artifacts", round(ghost_score,1)),
                (m6,"🔁","Copy","Duplication",     round(cm_score,1)),
                (m7,"🤖","AI","Pattern Detect",    round(ai_score,1)),
                (m8,"🎭","Mask","Suspicious Reg",  round(mask_score,1)),
            ]:
                with col:
                    st.markdown(f'<div class="metric-card"><div style="font-size:22px">{ic}</div><div class="metric-num" style="color:{sc(s)}">{s}%</div><div class="metric-name">{nm}</div><div class="metric-desc">{desc_t}</div></div>', unsafe_allow_html=True)
                    st.progress(min(int(s),100))

            st.markdown(f"""
            <div class="cs-divider">
                <div class="cs-divider-line"></div>
                <div class="cs-divider-text">{t["map_title"]}</div>
                <div class="cs-divider-line"></div>
            </div>
            """, unsafe_allow_html=True)

            ela_heat   = cv2.cvtColor(cv2.applyColorMap(ela_gray, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            ghost_heat = cv2.cvtColor(cv2.applyColorMap(ghost_map, cv2.COLORMAP_HOT), cv2.COLOR_BGR2RGB)
            cm_rgb     = cv2.cvtColor(cm_map, cv2.COLOR_GRAY2RGB)
            overlay    = cv2.cvtColor(apply_red_overlay(img_bgr, cv2.resize(mask_ref,(w,h))), cv2.COLOR_BGR2RGB)

            i1,i2,i3,i4,i5 = st.columns(5)
            with i1: st.image(img_rgb,    caption=t["img_orig"],     use_column_width=True)
            with i2: st.image(ela_heat,   caption=t["img_ela"],      use_column_width=True)
            with i3: st.image(ghost_heat, caption=t["img_ghost"],    use_column_width=True)
            with i4: st.image(cm_rgb,     caption=t["img_copymove"], use_column_width=True)
            with i5: st.image(overlay,    caption=t["img_mask"],     use_column_width=True)

            st.markdown(f'<div style="text-align:center;padding:12px;background:rgba(0,245,255,0.02);border-radius:10px;font-family:\'Share Tech Mono\',monospace;font-size:11px;color:#2a6a80;letter-spacing:2px;margin-top:10px">{t["map_legend"]}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(f"↺  {t['btn_reset']}", use_container_width=True):
            st.rerun()

# ════════════════════════════════════════════
#  ABOUT
# ════════════════════════════════════════════
elif st.session_state.page == "about":
    st.markdown(f"""
    <div style="text-align:center;padding:50px 0 30px">
        <div style="font-family:'Share Tech Mono',monospace;font-size:10px;color:#2a6a80;letter-spacing:5px;margin-bottom:12px">
            [ SYSTEM DOCUMENTATION ]
        </div>
        <div style="font-size:36px;font-weight:900;color:#c8f0ff;letter-spacing:2px">{t["about_title"]}</div>
    </div>
    """, unsafe_allow_html=True)

    a1,a2 = st.columns(2)
    cards = [
        ("🔬","#00f5ff","ELA" if not is_ar else "تقنية ELA",
         "Error Level Analysis detects tampered regions by comparing compression artifacts." if not is_ar else "تحلل مستوى الضغط وتكشف المناطق المعدلة بمقارنة بصمات الضغط."),
        ("📐","#b347ff","DCT Analysis" if not is_ar else "تحليل DCT",
         "Discrete Cosine Transform reveals abnormal coefficient patterns from editing." if not is_ar else "يكشف الانماط غير الطبيعية في معاملات التحويل الناتجة عن التعديل."),
        ("👻","#00ff88","JPEG Ghost" if not is_ar else "JPEG Ghost",
         "Detects regions compressed at different quality levels, revealing tampering." if not is_ar else "يكتشف الاجزاء المضغوطة بجودة مختلفة مما يدل على التعديل."),
        ("🔁","#ffaa00","Copy-Move" if not is_ar else "كشف Copy-Move",
         "ORB feature matching detects duplicated regions within the same document." if not is_ar else "يستخدم مطابقة النقاط المميزة لاكتشاف الاجزاء المنسوخة داخل الوثيقة."),
        ("🌊","#00f5ff","FFT Analysis" if not is_ar else "تحليل FFT",
         "Spatial frequency analysis reveals editing artifacts invisible to the naked eye." if not is_ar else "تحليل الترددات المكانية يكشف اثار التعديل غير المرئية للعين."),
        ("🤖","#ff3366","AI Detector" if not is_ar else "كاشف الذكاء الاصطناعي",
         "Grid artifact and noise pattern analysis for AI-generated content detection." if not is_ar else "يكشف المحتوى المولد بالذكاء الاصطناعي عبر تحليل انماط الشبكة والضوضاء."),
    ]
    for j, (ic,color,name,desc) in enumerate(cards):
        with [a1,a2][j%2]:
            st.markdown(f'<div class="about-card"><div style="font-size:26px;margin-bottom:10px">{ic}</div><div style="font-size:16px;font-weight:800;color:{color};margin-bottom:8px;letter-spacing:1px">{name}</div><div style="font-size:13px;color:#2a6a80;line-height:1.8">{desc}</div></div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="about-card" style="margin-top:8px">
        <div style="font-family:'Share Tech Mono',monospace;font-size:11px;color:#2a6a80;letter-spacing:3px;margin-bottom:20px">
            [ {"اوزان الحكم النهائي" if is_ar else "FINAL VERDICT WEIGHTS"} ]
        </div>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px;text-align:center">
            <div><div style="font-family:'Share Tech Mono',monospace;font-size:30px;font-weight:900;color:#00f5ff;filter:drop-shadow(0 0 8px #00f5ff)">25%</div><div style="font-size:11px;color:#2a6a80;letter-spacing:2px;margin-top:4px">ELA</div></div>
            <div><div style="font-family:'Share Tech Mono',monospace;font-size:30px;font-weight:900;color:#00ff88;filter:drop-shadow(0 0 8px #00ff88)">15%</div><div style="font-size:11px;color:#2a6a80;letter-spacing:2px;margin-top:4px">GHOST</div></div>
            <div><div style="font-family:'Share Tech Mono',monospace;font-size:30px;font-weight:900;color:#b347ff;filter:drop-shadow(0 0 8px #b347ff)">20%</div><div style="font-size:11px;color:#2a6a80;letter-spacing:2px;margin-top:4px">FFT</div></div>
            <div><div style="font-family:'Share Tech Mono',monospace;font-size:30px;font-weight:900;color:#ffaa00;filter:drop-shadow(0 0 8px #ffaa00)">10%</div><div style="font-size:11px;color:#2a6a80;letter-spacing:2px;margin-top:4px">COPY-MOVE</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════
#  CHAT
# ════════════════════════════════════════════
elif st.session_state.page == "chat":
    st.markdown(f"""
    <div style="text-align:center;padding:40px 0 20px">
        <div style="font-family:'Share Tech Mono',monospace;font-size:10px;color:#2a6a80;letter-spacing:5px;margin-bottom:12px">
            [ AI INVESTIGATOR ]
        </div>
        <div style="font-size:32px;font-weight:900;color:#c8f0ff;letter-spacing:1px">💬 {t["chat_title"]}</div>
        <div style="font-size:13px;color:#2a6a80;margin-top:8px;font-family:'Share Tech Mono',monospace;letter-spacing:1px">{t["chat_sub"]}</div>
    </div>
    """, unsafe_allow_html=True)

    chat_col, quick_col = st.columns([3,1])

    with quick_col:
        st.markdown(f"""
        <div style="background:var(--card);border:1px solid var(--border);border-radius:16px;padding:20px;margin-bottom:12px">
            <div style="font-family:'Share Tech Mono',monospace;font-size:10px;color:#2a6a80;letter-spacing:3px;margin-bottom:14px">
                [ {t["quick_q"]} ]
            </div>
        </div>
        """, unsafe_allow_html=True)
        for q in quick_questions[st.session_state.lang]:
            if st.button(q, key=f"qq_{q[:15]}", use_container_width=True):
                st.session_state.chat_history.append(("user", q))
                st.session_state.groq_history.append({"role":"user","content":q})
                with st.spinner(""):
                    try:
                        ans = chat(q, st.session_state.groq_history[:-1])
                    except:
                        ans = "عذراً، حدث خطأ. حاول مرة اخرى." if is_ar else "Sorry, an error occurred. Please try again."
                st.session_state.chat_history.append(("bot", ans))
                st.session_state.groq_history.append({"role":"assistant","content":ans})
                st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(t["chat_clear"], use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.groq_history = []
            st.rerun()

    with chat_col:
        msgs_html = ""
        for role, msg in st.session_state.chat_history:
            msg_d = msg.replace("\n", "<br>")
            if role == "user":
                msgs_html += f'<div class="msg-user">{msg_d}</div>'
            else:
                msgs_html += f'<div class="msg-bot"><div class="bot-label">▸ CERTISCAN AI</div>{msg_d}</div>'

        if not msgs_html:
            msgs_html = f"""
            <div style="text-align:center;padding:60px 20px;color:#1a4a5a">
                <div style="font-size:48px;margin-bottom:16px;filter:drop-shadow(0 0 20px rgba(0,245,255,0.3))">🤖</div>
                <div style="font-family:'Share Tech Mono',monospace;font-size:12px;letter-spacing:2px">
                    {"▸ جاهز للاجابة على اسئلتك" if is_ar else "▸ READY FOR QUERIES"}
                </div>
            </div>
            """

        st.markdown(f"""
        <div style="background:var(--card);border:1px solid var(--border);border-radius:20px;
             padding:24px;min-height:380px;direction:{dr};position:relative">
            <div class="corner-tl"></div><div class="corner-tr"></div>
            <div class="corner-bl"></div><div class="corner-br"></div>
            {msgs_html}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        user_input = st.text_input("", placeholder=t["chat_placeholder"], key="free_chat", label_visibility="collapsed")
        sc1, _ = st.columns([1,3])
        with sc1:
            if st.button(f"⟩ {t['chat_send']}", use_container_width=True):
                if user_input.strip():
                    st.session_state.chat_history.append(("user", user_input))
                    st.session_state.groq_history.append({"role":"user","content":user_input})
                    with st.spinner(""):
                        try:
                            ans = chat(user_input, st.session_state.groq_history[:-1])
                        except:
                            ans = "عذراً، حدث خطأ. حاول مرة اخرى." if is_ar else "Sorry, an error occurred. Please try again."
                    st.session_state.chat_history.append(("bot", ans))
                    st.session_state.groq_history.append({"role":"assistant","content":ans})
                    st.rerun()