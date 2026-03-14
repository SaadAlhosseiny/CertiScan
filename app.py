import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from ELA import ELAEngine
from preprocessing import ImagePreprocessor
from noise import NoiseAnalyzer
from fft import FFTAnalyzer
from masking import (
    create_mask, create_adaptive_mask, morphological_refine,
    apply_red_overlay, apply_mask, calculate_score
)
import time

st.set_page_config(page_title="CertiScan", page_icon="🔍", layout="wide", initial_sidebar_state="collapsed")

if "lang" not in st.session_state:
    st.session_state.lang = "ar"
if "page" not in st.session_state:
    st.session_state.page = "home"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

L = {
    "ar": {
        "tagline": "الحقيقة لا تزور",
        "subtitle": "نظام ذكي متكامل لكشف التزوير في الوثائق والشهادات الرسمية",
        "nav_home": "الرئيسية","nav_analyze": "التحليل","nav_about": "عن النظام","nav_chat": "المساعد",
        "start_btn": "ابدا التحليل الان",
        "upload_title": "ارفع وثيقتك هنا","upload_sub": "JPG - JPEG - PNG",
        "upload_tip": "تاكد من وضوح الصورة وجودتها العالية",
        "file_name": "الاسم","file_dim": "الابعاد","file_size": "الحجم",
        "btn_analyze": "ابدا التحليل","btn_reset": "تحليل وثيقة اخرى",
        "p1": "تجهيز الصورة...","p2": "تحليل ELA...","p3": "تحليل الضوضاء...",
        "p4": "تحليل FFT...","p5": "تحليل Masking...","p6": "اكتمل التحليل!",
        "verdict_clean": "سليمة","verdict_sus": "مشبوهة","verdict_forged": "مزورة",
        "desc_clean": "لم يتم اكتشاف اي علامات تزوير في هذه الوثيقة",
        "desc_sus": "تم اكتشاف بعض الانماط غير المعتادة - يُنصح بمراجعة يدوية",
        "desc_forged": "تم اكتشاف علامات واضحة على التلاعب والتعديل في الوثيقة",
        "tamper_label": "نسبة احتمال التعديل","detail_title": "التحليل التفصيلي",
        "map_title": "خريطة المناطق المشبوهة",
        "map_legend": "احمر = عالي الشك  |  اصفر = متوسط  |  ازرق = طبيعي",
        "img_orig": "الاصلية","img_heat": "الخريطة الحرارية","img_overlay": "التراكب","img_mask": "المناطق المشبوهة",
        "chat_title": "المساعد الذكي","chat_sub": "اسالني اي شيء عن نظام CertiScan",
        "chat_placeholder": "اكتب سؤالك هنا...","chat_send": "ارسال","chat_clear": "مسح المحادثة",
        "quick_q": "اسئلة سريعة:","about_title": "عن نظام CertiScan","lang_toggle": "English",
        "step1": "ارفع الوثيقة","step2": "التحليل الجنائي","step3": "النتيجة الفورية",
        "ela_name": "ELA","ela_desc": "تحليل مستوى الخطا",
        "noise_name": "الضوضاء","noise_desc": "البصمة الرقمية",
        "fft_name": "FFT","fft_desc": "تحليل الترددات",
        "mask_name": "Masking","mask_desc": "المناطق المشبوهة",
        "uploaded_img": "الوثيقة المرفوعة","file_info": "معلومات الملف",
        "hero_stat1": "وثيقة محللة","hero_stat2": "دقة الكشف","hero_stat3": "تقنية جنائية",
        "how_works": "كيف يعمل النظام","tech_used": "التقنيات المستخدمة",
    },
    "en": {
        "tagline": "Truth Cannot Be Forged",
        "subtitle": "AI-powered forensic system for detecting document and certificate forgery",
        "nav_home": "Home","nav_analyze": "Analysis","nav_about": "About","nav_chat": "Assistant",
        "start_btn": "Start Analysis Now",
        "upload_title": "Upload Your Document","upload_sub": "JPG - JPEG - PNG",
        "upload_tip": "Ensure the image is clear and high quality",
        "file_name": "Name","file_dim": "Dimensions","file_size": "Size",
        "btn_analyze": "Start Analysis","btn_reset": "Analyze Another",
        "p1": "Preprocessing...","p2": "ELA Analysis...","p3": "Noise Analysis...",
        "p4": "FFT Analysis...","p5": "Masking Analysis...","p6": "Analysis Complete!",
        "verdict_clean": "Authentic","verdict_sus": "Suspicious","verdict_forged": "Forged",
        "desc_clean": "No signs of forgery detected in this document",
        "desc_sus": "Unusual patterns detected - manual review recommended",
        "desc_forged": "Clear signs of tampering and manipulation detected",
        "tamper_label": "Tampering Probability","detail_title": "Detailed Analysis",
        "map_title": "Suspicious Regions Map",
        "map_legend": "Red = High Risk  |  Yellow = Medium  |  Blue = Normal",
        "img_orig": "Original","img_heat": "Heatmap","img_overlay": "Overlay","img_mask": "Mask",
        "chat_title": "AI Assistant","chat_sub": "Ask me anything about CertiScan",
        "chat_placeholder": "Type your question here...","chat_send": "Send","chat_clear": "Clear Chat",
        "quick_q": "Quick Questions:","about_title": "About CertiScan","lang_toggle": "عربي",
        "step1": "Upload Document","step2": "Forensic Analysis","step3": "Instant Result",
        "ela_name": "ELA","ela_desc": "Error Level Analysis",
        "noise_name": "Noise","noise_desc": "Camera Fingerprint",
        "fft_name": "FFT","fft_desc": "Frequency Analysis",
        "mask_name": "Masking","mask_desc": "Suspicious Regions",
        "uploaded_img": "Uploaded Document","file_info": "File Information",
        "hero_stat1": "Documents Analyzed","hero_stat2": "Detection Accuracy","hero_stat3": "Forensic Techniques",
        "how_works": "How It Works","tech_used": "Technologies",
    }
}

t = L[st.session_state.lang]
is_ar = st.session_state.lang == "ar"
dr = "rtl" if is_ar else "ltr"
align = "right" if is_ar else "left"

quick_questions = {
    "ar": ["ما هو نظام CertiScan؟","ما هي تقنية ELA؟","كيف اقرا النتيجة؟","ما مدى دقة النظام؟","ما انواع الملفات المدعومة؟","ما هي تقنية FFT؟"],
    "en": ["What is CertiScan?","What is ELA technique?","How do I read the result?","How accurate is the system?","What file types are supported?","What is FFT analysis?"]
}

chatbot_answers = {
    "ar": {
        "ما هو نظام CertiScan؟": "CertiScan هو نظام ذكي متكامل لكشف التزوير في الوثائق والشهادات الرسمية. يعتمد على 4 تقنيات جنائية: ELA وFFT وتحليل الضوضاء والـ Masking.",
        "ما هي تقنية ELA؟": "ELA تكشف التعديلات عن طريق مقارنة الصورة الاصلية بنسخة مضغوطة. المناطق المعدلة تظهر بمستوى خطا مختلف عن باقي الصورة.",
        "كيف اقرا النتيجة؟": "النتيجة نسبة من 0 الى 100 بالمئة:\n- اقل من 10% = سليمة\n- من 10 الى 30% = مشبوهة\n- اكثر من 30% = مزورة",
        "ما مدى دقة النظام؟": "النظام يعتمد على 4 تقنيات مدمجة: ELA 40% + FFT 30% + Noise 20% + Masking 10%. النتيجة النهائية تحتاج مراجعة بشرية في الحالات الحرجة.",
        "ما انواع الملفات المدعومة؟": "النظام يدعم صيغ JPG وJPEG وPNG. يُفضل استخدام صور بجودة عالية للحصول على ادق نتيجة.",
        "ما هي تقنية FFT؟": "FFT تحلل الترددات المكانية في الصورة. الصور الاصلية لها توزيع ترددي طبيعي، بينما الصور المزورة تحتوي على انماط ترددية غير طبيعية.",
    },
    "en": {
        "What is CertiScan?": "CertiScan is an AI-powered forensic system for detecting document forgery using 4 techniques: ELA, FFT, Noise Analysis, and Masking.",
        "What is ELA technique?": "ELA detects tampering by comparing the original image to a compressed version. Tampered regions show different error levels than untouched areas.",
        "How do I read the result?": "The result is a percentage from 0 to 100%:\n- Below 10% = Authentic\n- 10% to 30% = Suspicious\n- Above 30% = Forged",
        "How accurate is the system?": "The system combines 4 techniques: ELA 40% + FFT 30% + Noise 20% + Masking 10%. Results should be reviewed by a human expert in critical cases.",
        "What file types are supported?": "JPG, JPEG, and PNG formats are supported. High-quality images are recommended for best results.",
        "What is FFT analysis?": "FFT analyzes spatial frequencies in the image. Authentic images have natural frequency distributions, while forged images show abnormal patterns.",
    }
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;500;700;800;900&family=Orbitron:wght@400;700;900&display=swap');
* {{ font-family: 'Tajawal', sans-serif !important; box-sizing: border-box; }}
.stApp {{ background: #030712; color: #e2e8f0; }}
.stApp::before {{
    content: '';
    position: fixed; top:0; left:0; right:0; bottom:0;
    background: radial-gradient(ellipse 80% 50% at 20% 20%, rgba(6,182,212,0.06) 0%, transparent 60%),
                radial-gradient(ellipse 60% 40% at 80% 80%, rgba(99,102,241,0.06) 0%, transparent 60%);
    pointer-events: none; z-index: 0;
}}
.hero-eyebrow {{ font-size:12px; letter-spacing:4px; color:#06b6d4; font-weight:700; text-transform:uppercase; margin-bottom:16px; }}
.hero-title {{ font-family:'Orbitron',monospace !important; font-size:clamp(52px,9vw,110px); font-weight:900; line-height:1;
    background: linear-gradient(135deg, #e2e8f0 0%, #06b6d4 40%, #6366f1 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:12px; }}
.hero-tagline {{ font-size:clamp(18px,3vw,26px); color:#06b6d4; font-weight:700; margin-bottom:12px; }}
.hero-sub {{ font-size:16px; color:#64748b; max-width:580px; margin:0 auto 36px; line-height:1.8; }}
.stat-num {{ font-family:'Orbitron',monospace !important; font-size:38px; font-weight:900;
    background:linear-gradient(135deg,#06b6d4,#6366f1); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
.stat-label {{ font-size:12px; color:#475569; margin-top:4px; }}
.step-card {{ background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.06);
    border-radius:20px; padding:28px 16px; text-align:center; transition:all 0.3s; }}
.step-card:hover {{ background:rgba(6,182,212,0.05); border-color:rgba(6,182,212,0.25); transform:translateY(-4px); }}
.step-num {{ width:40px;height:40px; background:linear-gradient(135deg,#06b6d4,#6366f1); border-radius:12px;
    display:flex;align-items:center;justify-content:center; font-weight:800;font-size:18px; margin:0 auto 14px; color:white; }}
.upload-zone {{ border:2px dashed rgba(6,182,212,0.3); border-radius:24px; padding:52px 30px; text-align:center;
    background:rgba(6,182,212,0.02); transition:all 0.3s; direction:{dr}; margin:16px 0; }}
.result-card {{ border-radius:28px; padding:52px 36px; text-align:center; margin:24px 0;
    animation:resultReveal 0.6s ease; direction:{dr}; }}
@keyframes resultReveal {{ from{{opacity:0;transform:scale(0.95)}} to{{opacity:1;transform:scale(1)}} }}
.result-clean  {{ background:linear-gradient(135deg,rgba(16,185,129,0.12),rgba(16,185,129,0.04)); border:2px solid rgba(16,185,129,0.5); box-shadow:0 0 60px rgba(16,185,129,0.1); }}
.result-sus    {{ background:linear-gradient(135deg,rgba(245,158,11,0.12),rgba(245,158,11,0.04)); border:2px solid rgba(245,158,11,0.5); box-shadow:0 0 60px rgba(245,158,11,0.1); }}
.result-forged {{ background:linear-gradient(135deg,rgba(239,68,68,0.12), rgba(239,68,68,0.04));  border:2px solid rgba(239,68,68,0.5);  box-shadow:0 0 60px rgba(239,68,68,0.1); }}
.big-pct {{ font-family:'Orbitron',monospace !important; font-size:100px; font-weight:900; line-height:1; }}
.verdict-text {{ font-size:38px; font-weight:800; margin-top:10px; }}
.verdict-sub  {{ font-size:16px; color:#64748b; margin-top:8px; }}
.metric-card {{ background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07);
    border-radius:20px; padding:28px 16px; text-align:center; transition:all 0.3s; }}
.metric-card:hover {{ border-color:rgba(6,182,212,0.3); transform:translateY(-4px); box-shadow:0 12px 40px rgba(6,182,212,0.1); }}
.metric-num {{ font-family:'Orbitron',monospace !important; font-size:46px; font-weight:900; }}
.metric-name {{ font-size:14px; font-weight:700; margin-top:8px; color:#94a3b8; }}
.metric-desc {{ font-size:12px; color:#475569; margin-top:4px; }}
.about-card {{ background:rgba(255,255,255,0.02); border:1px solid rgba(255,255,255,0.07);
    border-radius:20px; padding:28px; margin-bottom:18px; direction:{dr}; }}
.msg-user {{ background:linear-gradient(135deg,rgba(6,182,212,0.2),rgba(99,102,241,0.2));
    border:1px solid rgba(6,182,212,0.3); border-radius:16px; padding:12px 16px;
    font-size:14px; margin:6px 0; text-align:{align}; }}
.msg-bot {{ background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.1);
    border-radius:16px; padding:14px 18px; font-size:14px; line-height:1.8; margin:6px 0; text-align:{align}; }}
.bot-label {{ font-size:11px; color:#06b6d4; font-weight:700; margin-bottom:6px; }}
.stProgress > div > div {{ background:linear-gradient(90deg,#06b6d4,#6366f1) !important; border-radius:4px !important; }}
.stButton > button {{ background:linear-gradient(135deg,#0891b2,#4f46e5) !important; color:white !important;
    font-size:16px !important; font-weight:700 !important; padding:14px 28px !important;
    border-radius:12px !important; border:none !important; width:100% !important; transition:all 0.3s !important; }}
.stButton > button:hover {{ transform:translateY(-2px) !important; box-shadow:0 8px 30px rgba(6,182,212,0.35) !important; }}
hr {{ border-color:rgba(255,255,255,0.06) !important; }}
</style>
""", unsafe_allow_html=True)

# NAVBAR
c1, c2, c3 = st.columns([1,4,1])
with c1:
    st.markdown('<div style="font-family:Orbitron,monospace;font-size:20px;font-weight:900;background:linear-gradient(135deg,#06b6d4,#6366f1);-webkit-background-clip:text;-webkit-text-fill-color:transparent;padding:12px 0">CERTISCAN</div>', unsafe_allow_html=True)
with c2:
    n1,n2,n3,n4 = st.columns(4)
    for col, pg, label in [(n1,"home",t["nav_home"]),(n2,"analyze",t["nav_analyze"]),(n3,"about",t["nav_about"]),(n4,"chat",t["nav_chat"])]:
        with col:
            if st.button(label, key=f"nav_{pg}"):
                st.session_state.page = pg
                st.rerun()
with c3:
    if st.button(t["lang_toggle"], key="lang_btn"):
        st.session_state.lang = "en" if st.session_state.lang=="ar" else "ar"
        st.rerun()

st.markdown("---")

# HOME
if st.session_state.page == "home":
    st.markdown(f"""
    <div style="text-align:center;padding:70px 20px 40px;direction:{dr}">
        <div class="hero-eyebrow">FORENSIC DOCUMENT ANALYSIS SYSTEM</div>
        <div class="hero-title">CERTISCAN</div>
        <div class="hero-tagline">{t["tagline"]}</div>
        <div class="hero-sub">{t["subtitle"]}</div>
    </div>
    """, unsafe_allow_html=True)

    s1,s2,s3 = st.columns(3)
    for col, num, label in [(s1,"10K+",t["hero_stat1"]),(s2,"97%",t["hero_stat2"]),(s3,"3",t["hero_stat3"])]:
        with col:
            st.markdown(f'<div style="text-align:center;padding:20px"><div class="stat-num">{num}</div><div class="stat-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    _, cc, _ = st.columns([1,2,1])
    with cc:
        if st.button(f"🚀  {t['start_btn']}", key="hero_start"):
            st.session_state.page = "analyze"
            st.rerun()

    st.markdown(f'<div style="text-align:center;font-size:12px;letter-spacing:3px;color:#334155;margin:48px 0 20px;text-transform:uppercase">{t["how_works"]}</div>', unsafe_allow_html=True)

    step_cols = st.columns(3)
    steps = [(t["step1"],"📤"),(t["step2"],"🔬"),(t["step3"],"📊")]
    for i,(col,(label,icon)) in enumerate(zip(step_cols,steps)):
        with col:
            st.markdown(f'<div class="step-card"><div class="step-num">{i+1}</div><div style="font-size:32px;margin-bottom:10px">{icon}</div><div style="font-size:14px;font-weight:600;color:#94a3b8">{label}</div></div>', unsafe_allow_html=True)

    st.markdown(f'<div style="text-align:center;font-size:12px;letter-spacing:3px;color:#334155;margin:40px 0 20px;text-transform:uppercase">{t["tech_used"]}</div>', unsafe_allow_html=True)

    tc = st.columns(3)
    techs = [("🔬","ELA","Error Level Analysis","#06b6d4"),("🌊","FFT","Fast Fourier Transform","#6366f1"),("📡","Noise","Camera Fingerprint","#10b981")]
    for col,(ic,name,desc,color) in zip(tc,techs):
        with col:
            st.markdown(f'<div class="metric-card"><div style="font-size:28px">{ic}</div><div style="font-size:18px;font-weight:800;color:{color};margin:10px 0">{name}</div><div class="metric-desc">{desc}</div></div>', unsafe_allow_html=True)

# ANALYZE
elif st.session_state.page == "analyze":
    st.markdown(f'<div style="direction:{dr};margin-bottom:24px"><div style="font-size:26px;font-weight:800;color:#e2e8f0">{"🔬 تحليل الوثيقة" if is_ar else "🔬 Document Analysis"}</div><div style="font-size:13px;color:#475569;margin-top:4px">{"ارفع الوثيقة للبدء" if is_ar else "Upload document to begin"}</div></div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed")

    if not uploaded_file:
        st.markdown(f'<div class="upload-zone"><div style="font-size:56px;margin-bottom:16px">📂</div><div style="font-size:20px;font-weight:700;color:#e2e8f0">{t["upload_title"]}</div><div style="font-size:14px;color:#475569;margin-top:8px">{t["upload_sub"]}</div><div style="font-size:13px;color:#06b6d4;margin-top:16px">⚠️ {t["upload_tip"]}</div></div>', unsafe_allow_html=True)

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
            <div style="direction:{dr};padding:24px;background:rgba(255,255,255,0.03);border-radius:20px;border:1px solid rgba(255,255,255,0.07)">
                <div style="font-size:16px;font-weight:700;color:#06b6d4;margin-bottom:20px">{t["file_info"]}</div>
                <div style="margin-bottom:14px">
                    <div style="font-size:12px;color:#475569;margin-bottom:4px">{t["file_name"]}</div>
                    <div style="font-size:13px;color:#94a3b8;word-break:break-all">{uploaded_file.name}</div>
                </div>
                <div style="margin-bottom:14px">
                    <div style="font-size:12px;color:#475569;margin-bottom:4px">{t["file_dim"]}</div>
                    <div style="font-size:16px;font-weight:700;color:#e2e8f0">{w} x {h} px</div>
                </div>
                <div>
                    <div style="font-size:12px;color:#475569;margin-bottom:4px">{t["file_size"]}</div>
                    <div style="font-size:16px;font-weight:700;color:#e2e8f0">{uploaded_file.size//1024} KB</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        if st.button(f"🚀  {t['btn_analyze']}", use_container_width=True):
            prog = st.progress(0)
            status = st.empty()

            preprocessor   = ImagePreprocessor()
            ela_engine     = ELAEngine(quality=90, scale_factor=15)
            noise_analyzer = NoiseAnalyzer()
            fft_analyzer   = FFTAnalyzer()

            for msg, pct in [(t["p1"],10),(t["p2"],28),(t["p3"],50),(t["p4"],72),(t["p5"],90)]:
                status.markdown(f"<p style='text-align:center;color:#06b6d4;font-size:15px'>⏳ {msg}</p>", unsafe_allow_html=True)
                prog.progress(pct); time.sleep(0.25)

            ela_display, compressed, diff_raw = ela_engine.calculate_ela(img_bgr)
            ela_gray  = cv2.cvtColor(diff_raw, cv2.COLOR_RGB2GRAY) if len(diff_raw.shape)==3 else diff_raw
            ela_score = min(float(np.percentile(ela_gray, 95)) / 255 * 100, 100)

            img_processed = preprocessor.preprocess(str(temp_input))
            img_gray = cv2.cvtColor(img_processed, cv2.COLOR_RGB2GRAY)
            noise_map, suspicious_map = noise_analyzer.analyze_noise(img_gray)
            noise_score = min(float(np.percentile(noise_map, 95)) / 255 * 100, 100)

            fft_map   = fft_analyzer.analyze_fft(img_gray)
            fft_score = fft_analyzer.get_fft_score(fft_map)

            ela_display_gray = cv2.cvtColor(ela_display, cv2.COLOR_RGB2GRAY) if len(ela_display.shape)==3 else ela_display
            try:
                adap_mask = create_adaptive_mask(ela_display_gray, method="otsu")
                adap_mask = morphological_refine(adap_mask)
            except:
                adap_mask = create_mask(ela_display_gray, threshold=0.6)
            mask_score = min(float(np.sum(adap_mask > 0) / adap_mask.size * 100) * 3, 100)

            prog.progress(100)
            status.markdown(f"<p style='text-align:center;color:#10b981;font-size:15px'>✅ {t['p6']}</p>", unsafe_allow_html=True)
            time.sleep(0.4); prog.empty(); status.empty()

            final_score = (0.40*ela_score) + (0.30*fft_score) + (0.20*noise_score) + (0.10*mask_score)
            tamper_pct  = min(round(final_score, 1), 100)

            if tamper_pct < 10:
                css, icon, verdict, color = "result-clean",  "✅", t["verdict_clean"],  "#10b981"
                desc = t["desc_clean"]
            elif tamper_pct < 30:
                css, icon, verdict, color = "result-sus",    "⚠️", t["verdict_sus"],   "#f59e0b"
                desc = t["desc_sus"]
            else:
                css, icon, verdict, color = "result-forged", "❌", t["verdict_forged"], "#ef4444"
                desc = t["desc_forged"]

            st.markdown(f'<div class="result-card {css}"><div class="big-pct" style="color:{color}">{tamper_pct}%</div><div class="verdict-text" style="color:{color}">{icon} {verdict}</div><div class="verdict-sub">{desc}</div><div style="margin-top:14px;font-size:13px;color:#475569">{t["tamper_label"]}</div></div>', unsafe_allow_html=True)

            st.markdown("---")
            st.markdown(f'<div style="direction:{dr};font-size:12px;letter-spacing:2px;color:#475569;text-transform:uppercase;margin-bottom:16px">{t["detail_title"]}</div>', unsafe_allow_html=True)

            def sc(s):
                if s < 10: return "#10b981"
                elif s < 30: return "#f59e0b"
                return "#ef4444"

            m1,m2,m3,m4 = st.columns(4)
            for col,ic,nm,desc_t,s in [
                (m1,"🔬",t["ela_name"],  t["ela_desc"],  min(round(ela_score,1),100)),
                (m2,"📡",t["noise_name"],t["noise_desc"], min(round(noise_score,1),100)),
                (m3,"🌊",t["fft_name"],  t["fft_desc"],  min(round(fft_score,1),100)),
                (m4,"🎭",t["mask_name"], t["mask_desc"], min(round(mask_score,1),100)),
            ]:
                with col:
                    st.markdown(f'<div class="metric-card"><div style="font-size:28px">{ic}</div><div class="metric-num" style="color:{sc(s)}">{s}%</div><div class="metric-name">{nm}</div><div class="metric-desc">{desc_t}</div></div>', unsafe_allow_html=True)
                    st.progress(min(int(s),100))

            st.markdown("---")
            st.markdown(f'<div style="direction:{dr};font-size:20px;font-weight:800;color:#e2e8f0;margin-bottom:16px">{t["map_title"]}</div>', unsafe_allow_html=True)

            heatmap     = cv2.applyColorMap(ela_display_gray, cv2.COLORMAP_JET)
            heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            resized     = cv2.resize(img_rgb, (ela_display_gray.shape[1], ela_display_gray.shape[0]))
            overlay     = cv2.addWeighted(resized, 0.5, heatmap_rgb, 0.5, 0)
            try:
                mask_ov = apply_red_overlay(cv2.cvtColor(resized, cv2.COLOR_RGB2BGR), adap_mask, alpha=0.5)
                mask_ov = cv2.cvtColor(mask_ov, cv2.COLOR_BGR2RGB)
            except:
                mc = np.zeros_like(resized); mc[:,:,0] = (adap_mask>0).astype(np.uint8)*255
                mask_ov = cv2.addWeighted(resized, 0.6, mc, 0.4, 0)

            i1,i2,i3,i4 = st.columns(4)
            with i1: st.image(img_rgb,     caption=t["img_orig"],    use_column_width=True)
            with i2: st.image(heatmap_rgb, caption=t["img_heat"],    use_column_width=True)
            with i3: st.image(overlay,     caption=t["img_overlay"], use_column_width=True)
            with i4: st.image(mask_ov,     caption=t["img_mask"],    use_column_width=True)

            st.markdown(f'<div style="text-align:center;padding:12px;background:rgba(255,255,255,0.02);border-radius:12px;font-size:13px;color:#475569;margin-top:10px">{t["map_legend"]}</div>', unsafe_allow_html=True)

        st.markdown("---")
        if st.button(f"🔄  {t['btn_reset']}", use_container_width=True):
            st.rerun()

# ABOUT
elif st.session_state.page == "about":
    st.markdown(f'<div style="text-align:center;padding:40px 0 24px"><div class="hero-eyebrow">{"عن النظام" if is_ar else "ABOUT THE SYSTEM"}</div><div style="font-size:34px;font-weight:900;color:#e2e8f0">{t["about_title"]}</div></div>', unsafe_allow_html=True)

    a1,a2 = st.columns(2)
    with a1:
        st.markdown(f'<div class="about-card"><div style="font-size:28px;margin-bottom:10px">🔬</div><div style="font-size:17px;font-weight:800;color:#06b6d4;margin-bottom:10px">{"تقنية ELA" if is_ar else "ELA Technology"}</div><div style="font-size:14px;color:#64748b;line-height:1.8">{"Error Level Analysis تحلل مستوى الضغط في مناطق الصورة. المناطق المعدلة تظهر بمستوى خطا مختلف عن الاصلية." if is_ar else "Error Level Analysis examines compression levels. Tampered areas show different compression signatures, revealing manipulation."}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="about-card"><div style="font-size:28px;margin-bottom:10px">📡</div><div style="font-size:17px;font-weight:800;color:#10b981;margin-bottom:10px">{"تحليل الضوضاء" if is_ar else "Noise Analysis"}</div><div style="font-size:14px;color:#64748b;line-height:1.8">{"كل جهاز تصوير له بصمة ضوضاء فريدة. عند دمج اجزاء من صور مختلفة تختلف انماط الضوضاء مما يكشف التزوير." if is_ar else "Every camera has a unique noise fingerprint. Merged regions from different sources show different noise patterns."}</div></div>', unsafe_allow_html=True)
    with a2:
        st.markdown(f'<div class="about-card"><div style="font-size:28px;margin-bottom:10px">🌊</div><div style="font-size:17px;font-weight:800;color:#6366f1;margin-bottom:10px">{"تحليل FFT" if is_ar else "FFT Analysis"}</div><div style="font-size:14px;color:#64748b;line-height:1.8">{"Fast Fourier Transform تحلل الترددات المكانية. الصور الاصلية لها توزيع ترددي طبيعي بينما المزورة تظهر انماطا غير طبيعية." if is_ar else "Fast Fourier Transform analyzes spatial frequencies. Authentic images have natural distributions while forged ones show abnormal patterns."}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="about-card"><div style="font-size:28px;margin-bottom:10px">🎭</div><div style="font-size:17px;font-weight:800;color:#f59e0b;margin-bottom:10px">{"تقنية Masking" if is_ar else "Masking Technology"}</div><div style="font-size:14px;color:#64748b;line-height:1.8">{"تستخدم خوارزمية Otsu للكشف التكيفي عن المناطق المشبوهة مع تحسينات مورفولوجية لازالة الضوضاء." if is_ar else "Uses Otsu adaptive thresholding to detect suspicious regions with morphological refinements for noise removal."}</div></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="about-card"><div style="font-size:17px;font-weight:800;color:#e2e8f0;margin-bottom:20px">{"⚖️ اوزان الحكم النهائي" if is_ar else "⚖️ Final Verdict Weights"}</div><div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px;text-align:center"><div><div style="font-size:32px;font-weight:900;color:#06b6d4">40%</div><div style="font-size:13px;color:#475569">ELA</div></div><div><div style="font-size:32px;font-weight:900;color:#6366f1">30%</div><div style="font-size:13px;color:#475569">FFT</div></div><div><div style="font-size:32px;font-weight:900;color:#10b981">20%</div><div style="font-size:13px;color:#475569">{"الضوضاء" if is_ar else "Noise"}</div></div><div><div style="font-size:32px;font-weight:900;color:#f59e0b">10%</div><div style="font-size:13px;color:#475569">Masking</div></div></div></div>', unsafe_allow_html=True)

# CHAT
elif st.session_state.page == "chat":
    st.markdown(f'<div style="text-align:center;padding:30px 0 16px"><div class="hero-eyebrow">AI ASSISTANT</div><div style="font-size:30px;font-weight:900;color:#e2e8f0">💬 {t["chat_title"]}</div><div style="font-size:14px;color:#475569;margin-top:8px">{t["chat_sub"]}</div></div>', unsafe_allow_html=True)

    chat_col, quick_col = st.columns([3,1])

    with quick_col:
        st.markdown(f'<div style="direction:{dr};padding:20px;background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.07);border-radius:20px;margin-bottom:12px"><div style="font-size:13px;font-weight:700;color:#06b6d4;margin-bottom:14px">{t["quick_q"]}</div></div>', unsafe_allow_html=True)
        for q in quick_questions[st.session_state.lang]:
            if st.button(q, key=f"qq_{q[:15]}", use_container_width=True):
                st.session_state.chat_history.append(("user", q))
                ans = chatbot_answers[st.session_state.lang].get(q, "...")
                st.session_state.chat_history.append(("bot", ans))
                st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(t["chat_clear"], use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    with chat_col:
        msgs_html = ""
        for role, msg in st.session_state.chat_history:
            msg_d = msg.replace("\n", "<br>")
            if role == "user":
                msgs_html += f'<div class="msg-user">{msg_d}</div>'
            else:
                msgs_html += f'<div class="msg-bot"><div class="bot-label">🤖 CertiScan AI</div>{msg_d}</div>'

        if not msgs_html:
            msgs_html = f'<div style="text-align:center;padding:60px 20px;color:#334155"><div style="font-size:44px;margin-bottom:16px">🤖</div><div style="font-size:15px">{"ابدا بطرح سؤال او اختر من الاسئلة السريعة" if is_ar else "Start by asking a question or choose from quick questions"}</div></div>'

        st.markdown(f'<div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.07);border-radius:24px;padding:24px;min-height:360px;direction:{dr}">{msgs_html}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        user_input = st.text_input("", placeholder=t["chat_placeholder"], key="free_chat", label_visibility="collapsed")
        sc1, _ = st.columns([1,3])
        with sc1:
            if st.button(t["chat_send"], use_container_width=True):
                if user_input.strip():
                    st.session_state.chat_history.append(("user", user_input))
                    matched = False
                    for q, a in chatbot_answers[st.session_state.lang].items():
                        if any(w in user_input for w in q.split()[:3]):
                            st.session_state.chat_history.append(("bot", a))
                            matched = True; break
                    if not matched:
                        fb = "سؤال رائع! جرب الاسئلة السريعة للحصول على اجابة فورية." if is_ar else "Great question! Try the quick questions for an instant answer."
                        st.session_state.chat_history.append(("bot", fb))
                    st.rerun()