import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai
import json
from PIL import Image
import time
import sys
import os
sys.path.append(os.getcwd())
try:
    import CoolProp.CoolProp as CP
except ImportError:
    st.error("CoolProp 라이브러리가 설치되지 않았습니다. requirements.txt를 확인해 주세요.")
    st.stop()

# =============================================================================
# [내장형 물성치 모듈]
# =============================================================================
FLUID_OPTIONS = {
    "Water"            : "물 (Water)",
    "Methanol"         : "메탄올 (Methanol)",
    "Ethanol"          : "에탄올 (Ethanol)",
    "INCOMP::MEG[0.5]" : "에틸렌 글리콜 (부동액 50% 수용액)",
    "INCOMP::MPG[0.5]" : "프로필렌 글리콜 (부동액 50% 수용액)",
    "Acetone"          : "아세톤 (Acetone)",
    "Benzene"          : "벤젠 (Benzene)",
    "Toluene"          : "톨루엔 (Toluene)",
}

ROUGHNESS = {
    "Smooth Pipe (초매끈한 관, ε=0)": 0.0,
    "PVC (일반 플라스틱 관)": 1.5e-6,
    "Commercial Steel (상업용 강관)": 4.6e-5,
    "Galvanized Steel (아연도금 강관)": 1.5e-4,
    "Cast Iron (주철관)": 2.6e-4,
    "Concrete (콘크리트관)": 1.5e-3,
    "Drawn Tubing (인발 튜브)": 1.5e-6,
    "Stainless Steel (스테인리스 강관)": 1.5e-5,
}

def get_fluid_properties(fluid: str, temp_c: float) -> tuple:
    T_K = temp_c + 273.15
    P = 101325.0
    p_vapor = 0.0
    if fluid.startswith("INCOMP::"):
        rho = CP.PropsSI("D", "T", T_K, "P", P, fluid)
        mu  = CP.PropsSI("V", "T", T_K, "P", P, fluid)
    else:
        try:
            rho = CP.PropsSI("D", "T", T_K, "Q", 0, fluid)
            mu  = CP.PropsSI("V", "T", T_K, "Q", 0, fluid)
        except ValueError:
            rho = CP.PropsSI("D", "T", T_K, "P", P, fluid)
            mu  = CP.PropsSI("V", "T", T_K, "P", P, fluid)
    return rho, mu, p_vapor

def calc_friction_factor(Re: float, D: float, epsilon: float) -> tuple:
    if Re < 1e-6:
        return 0.0, "정지 (No Flow)"
    if Re < 2300:
        f = 64.0 / Re
        regime = "층류 (Laminar)"
    else:
        D = max(D, 1e-9)
        relative_roughness = epsilon / D
        denom = np.log10(relative_roughness / 3.7 + 5.74 / (Re**0.9))
        f = 0.25 / denom**2
        regime = "난류 (Turbulent)"
    return f, regime
# =============================================================================
# 배관망 해석기 (Hardy Cross 법) & AI 도면 인식
# =============================================================================

st.set_page_config(page_title="배관망 해석기 (Hardy Cross)", page_icon="🌀", layout="wide")

# 커스텀 CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0F2027 0%, #203A43 50%, #2C5364 100%);
        padding: 2.5rem; border-radius: 16px; color: white; margin-bottom: 2rem;
        text-align: center; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 { margin: 0; font-size: 2.5rem; color: white; font-weight: 800;}
    .main-header p  { margin-top: 0.5rem; opacity: 0.9; font-size: 1.1rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🌀 AI 기반 배관망 유동 해석기 (Hardy Cross Method)</h1>
    <p>도면 이미지를 업로드하면 AI가 배관 구조를 분석하고, 하디크로스 법을 통해 유량을 도출합니다.</p>
</div>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if "network_data" not in st.session_state:
    st.session_state["network_data"] = None
if "pipes_df" not in st.session_state:
    st.session_state["pipes_df"] = None
if "loops_df" not in st.session_state:
    st.session_state["loops_df"] = None

# =============================================================================
# [사이드바] 환경 설정
# =============================================================================
with st.sidebar:
    st.header("⚙️ 기본 설정")
    
    # API 키를 secrets에서 자동으로 가져오며, 화면에는 노출하지 않습니다.
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        st.success("✅ Gemini AI 자동 연동 완료 (보안 키 적용됨)")
    except Exception:
        api_key = ""
        st.error("❌ 서버에 Gemini API Key가 설정되지 않았습니다.")
    
    st.markdown("---")
    st.subheader("🧮 하디크로스 해석 변수")
    head_loss_eq = st.selectbox(
        "손실 수두 공식 선택",
        options=["Darcy-Weisbach (n=2.0)", "Hazen-Williams (n=1.852)"]
    )
    n_exp = 2.0 if "Darcy" in head_loss_eq else 1.852
    
    st.markdown("---")
    max_iter = st.number_input("최대 반복 횟수 (Iterations)", min_value=1, max_value=1000, value=100, step=10)
    tolerance = st.number_input("허용 오차 (Tolerance, ΔQ)", min_value=0.0001, max_value=0.1, value=0.001, step=0.0001, format="%.4f")

    st.markdown("---")
    st.subheader("💧 유체 및 배관 물성치 상세 설정")
    
    fluid_list = list(FLUID_OPTIONS.keys())
    fluid_display = [FLUID_OPTIONS[f] for f in fluid_list]
    fluid_choice = st.selectbox("유체 종류", fluid_display)
    fluid_type = fluid_list[fluid_display.index(fluid_choice)]
    
    fluid_temp = st.number_input("유체 온도 (°C)", min_value=-50.0, max_value=300.0, value=20.0, step=1.0)
    
    material_list = list(ROUGHNESS.keys())
    def_idx = 2 if len(material_list) > 2 else 0
    pipe_material = st.selectbox("배관 재질 (조도 ε 결정)", material_list, index=def_idx)

# =============================================================================
# 1. AI 도면 인식 섹션
# =============================================================================
st.subheader("🖼️ 1. 평면도 이미지 업로드 및 AI 인식")
col_img, col_ai = st.columns([1, 1.5])

uploaded_file = None
with col_img:
    st.info("배관망 평면도(이미지)를 업로드하세요. JPG, PNG 파일을 지원합니다.")
    uploaded_file = st.file_uploader("이미지 업로드", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="업로드된 도면", use_column_width=True)

with col_ai:
    st.markdown("**🤖 Gemini AI 자동 추출**")
    st.write("이미지에 포함된 각 루프별 배관 ID, K(저항계수), 초기 가상 유량(Q0)을 추출합니다.")
    
    analyze_btn = st.button("🚀 AI 분석 시작", type="primary", use_container_width=True, disabled=(uploaded_file is None))
    
    if analyze_btn:
        if not api_key:
            st.error("왼쪽 사이드바에서 Gemini API Key를 입력해주세요.")
        else:
            with st.spinner("AI가 도면을 분석 중입니다... (약 10~20초 소요)"):
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-flash-latest')
                    
                    prompt = """
                    당신은 수리학 및 배관망 해석(Hardy Cross Method) 전문가입니다.
                    제공된 배관망 평면도 이미지를 분석하여 다음 정보를 JSON 형식으로 추출해 주세요.
                    
                    추출할 데이터 구조는 다음과 같아야 합니다 (반드시 유효한 JSON 형식만 출력하세요, 마크다운 코드 블록 제외):
                    {
                        "pipes": [
                            {"pipe_id": "1", "D_m": 0.25, "L_m": 300.0, "Q_initial": 0.05},
                            {"pipe_id": "2", "D_m": 0.20, "L_m": 250.0, "Q_initial": 0.02}
                        ],
                        "loops": [
                            {
                                "loop_id": "I",
                                "pipes_in_loop": [
                                    {"pipe_id": "1", "direction": 1},
                                    {"pipe_id": "2", "direction": -1}
                                ]
                            }
                        ]
                    }
                    
                    규칙:
                    1. pipes 배열에는 배관 ID(pipe_id), 직경(D_m, 미터 단위), 길이(L_m, 미터 단위), 가상 초기 유량(Q_initial)이 포함되어야 합니다. (도면에 D=0.25m, L=300m 처럼 적혀있습니다.)
                    2. 도면에 주어진 외부 유입/유출량을 분석하여, 모든 노드(분기점)에서 '들어오는 유량 = 나가는 유량'이 되도록 **Q_initial 값을 합리적으로 추정**하세요. (초기 유량이 연속방정식을 만족해야 해석이 가능합니다.)
                    3. loops 배열에는 폐회로(루프) 정보가 들어갑니다.
                    4. direction은 해당 루프의 시계방향(Clockwise)을 기준으로, 배관의 초기 유동 방향이 시계방향과 일치하면 1, 반대면 -1을 넣습니다.
                    5. 도면을 최대한 파악해서 모든 파이프(1~7번 등)와 루프(I, II 등)를 찾아내세요.
                    """
                    
                    response = model.generate_content([prompt, image])
                    text_resp = response.text
                    
                    # JSON 파싱을 위해 마크다운 블록 제거
                    text_resp = text_resp.replace('```json', '').replace('```', '').strip()
                    extracted_data = json.loads(text_resp)
                    
                    st.session_state["network_data"] = extracted_data
                    
                    # DataFrame으로 변환
                    pipes_list = extracted_data.get("pipes", [])
                    st.session_state["pipes_df"] = pd.DataFrame(pipes_list)
                    
                    loop_list = []
                    for l in extracted_data.get("loops", []):
                        for p in l.get("pipes_in_loop", []):
                            loop_list.append({
                                "loop_id": l["loop_id"],
                                "pipe_id": p["pipe_id"],
                                "direction": p["direction"]
                            })
                    st.session_state["loops_df"] = pd.DataFrame(loop_list)
                    
                    st.success("데이터 추출 성공!")
                except Exception as e:
                    st.error(f"AI 분석 중 오류가 발생했습니다: {e}")
                    st.warning("JSON 형식이 아니거나 이미지를 인식하지 못했을 수 있습니다. 아래에서 수동으로 데이터를 입력할 수 있습니다.")


# =============================================================================
# 2. 데이터 수동 입력 및 편집
# =============================================================================
st.markdown("---")
st.subheader("📝 2. 배관망 데이터 검토 및 수정")
st.info("AI가 추출한 데이터를 확인하거나, 수동으로 데이터를 직접 입력/수정할 수 있습니다.")

# 기본 데이터 세팅
if st.session_state["pipes_df"] is None:
    st.session_state["pipes_df"] = pd.DataFrame({
        "pipe_id": ["1", "2", "3", "4"],
        "D_m": [0.25, 0.20, 0.20, 0.25],
        "L_m": [300.0, 250.0, 300.0, 250.0],
        "Q_initial": [0.05, 0.02, 0.03, -0.01]
    })
if st.session_state["loops_df"] is None:
    st.session_state["loops_df"] = pd.DataFrame({
        "loop_id": ["I", "I", "I", "I"],
        "pipe_id": ["1", "2", "3", "4"],
        "direction": [1, 1, -1, -1]
    })

col_d1, col_d2 = st.columns(2)

with col_d1:
    st.markdown("**배관 기본 정보 (Pipes)**")
    edited_pipes = st.data_editor(
        st.session_state["pipes_df"], 
        num_rows="dynamic", 
        use_container_width=True,
        key="editor_pipes"
    )

with col_d2:
    st.markdown("**루프 구성 (Loops)**")
    edited_loops = st.data_editor(
        st.session_state["loops_df"], 
        num_rows="dynamic", 
        use_container_width=True,
        key="editor_loops"
    )


# =============================================================================
# 3. Hardy Cross 연산
# =============================================================================
st.markdown("---")
st.subheader("🔄 3. 하디크로스 수치해석 (Hardy Cross Method)")

calc_btn = st.button("🚀 유량 해석 실행", type="primary")

if calc_btn:
    try:
        # 데이터 준비
        pipes_data = edited_pipes.set_index("pipe_id").to_dict(orient="index")
        
        loops_dict = {}
        for _, row in edited_loops.iterrows():
            l_id = row["loop_id"]
            p_id = row["pipe_id"]
            dir_val = row["direction"]
            if l_id not in loops_dict:
                loops_dict[l_id] = []
            loops_dict[l_id].append({"id": p_id, "dir": dir_val})
            
        # 초기 유량 세팅
        Q_current = {pid: float(info["Q_initial"]) for pid, info in pipes_data.items()}
        
        st.write(f"🔹 **적용 공식**: {head_loss_eq} (n={n_exp})")
        st.info("💡 파이프의 직경(D)과 길이(L)를 바탕으로 매 반복마다 마찰계수(f)와 저항계수(K)를 동적으로 정밀하게 계산합니다.")
        
        # 반복 연산 수행
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        iteration = 0
        converged = False
        
        # 물성치 준비
        rho = 998.2
        mu = 0.001002
        epsilon = 4.6e-5 # 기본 강관 조도
        try: 
            rho, mu, _ = get_fluid_properties(fluid_type, fluid_temp)
            epsilon = ROUGHNESS.get(pipe_material, 4.6e-5)
        except: pass
        
        for iteration in range(1, int(max_iter) + 1):
            max_delta = 0.0
            
            # 루프별 연산
            for l_id, p_list in loops_dict.items():
                numerator = 0.0
                denominator = 0.0
                
                for p in p_list:
                    pid = p["id"]
                    direction = p["dir"]
                    
                    if pid not in Q_current:
                        continue
                        
                    Q_loop = Q_current[pid] * direction
                    abs_Q = abs(Q_loop)
                    
                    D_m = float(pipes_data[pid]["D_m"])
                    L_m = float(pipes_data[pid]["L_m"])
                    
                    # K값 동적 계산
                    if n_exp == 2.0: # Darcy-Weisbach
                        if abs_Q < 1e-6:
                            f_val = 0.02
                        else:
                            v = (4.0 * abs_Q) / (np.pi * D_m**2)
                            Re_val = (rho * v * D_m) / mu
                            f_val, _ = calc_friction_factor(Re_val, D_m, epsilon)
                        K_val = (8.0 * f_val * L_m) / (np.pi**2 * 9.81 * D_m**5)
                    else: # Hazen-Williams
                        C = 130.0
                        K_val = (10.67 * L_m) / ((C**1.852) * (D_m**4.87))
                    
                    # 분자: sum( K * Q * |Q|^(n-1) )
                    term_num = K_val * Q_loop * (abs_Q**(n_exp - 1))
                    numerator += term_num
                    
                    # 분모: sum( n * K * |Q|^(n-1) )
                    term_den = n_exp * K_val * (abs_Q**(n_exp - 1))
                    denominator += term_den
                
                # Delta Q 계산
                if denominator == 0:
                    delta_Q = 0
                else:
                    delta_Q = - (numerator / denominator)
                
                # 최대 오차 갱신
                if abs(delta_Q) > max_delta:
                    max_delta = abs(delta_Q)
                    
                # 유량 업데이트
                for p in p_list:
                    pid = p["id"]
                    direction = p["dir"]
                    Q_current[pid] += delta_Q * direction
            
            # 진행 상태 업데이트
            progress_bar.progress(min(iteration / max_iter, 1.0))
            status_text.text(f"Iteration {iteration} 진행 중... 최대 ΔQ = {max_delta:.6f}")
            
            # 수렴 판정
            if max_delta < tolerance:
                converged = True
                break
                
        time.sleep(0.5)
        
        if converged:
            st.success(f"🎉 해석 완료! 총 {iteration}회 반복 후 수렴에 도달했습니다. (최대 오차: {max_delta:.6f})")
        else:
            st.warning(f"⚠️ 해석 완료. {max_iter}회 반복 동안 수렴하지 않았습니다. (마지막 최대 오차: {max_delta:.6f})")
        
        # 결과 DataFrame 생성 (교과서 5단원 양식 반영)
        res_list = []
        
        for l_id, p_list in loops_dict.items():
            for p in p_list:
                pid = p["id"]
                direction = p["dir"]
                
                # 해당 루프에서의 방향을 고려한 최종 유량
                Q_final_loop = Q_current[pid] * direction
                abs_Q = abs(Q_final_loop)
                
                D_m = float(pipes_data[pid]["D_m"])
                L_m = float(pipes_data[pid]["L_m"])
                
                Re_val = 0.0
                f_val = 0.0
                
                if n_exp == 2.0:
                    if abs_Q < 1e-6:
                        f_val = 0.02
                    else:
                        v = (4.0 * abs_Q) / (np.pi * D_m**2)
                        Re_val = (rho * v * D_m) / mu
                        f_val, _ = calc_friction_factor(Re_val, D_m, epsilon)
                    K_val = (8.0 * f_val * L_m) / (np.pi**2 * 9.81 * D_m**5)
                else:
                    C = 130.0
                    K_val = (10.67 * L_m) / ((C**1.852) * (D_m**4.87))
                
                # 압력강하(hf) 및 hf/Q 계산
                hf = K_val * Q_final_loop * (abs_Q**(n_exp - 1))
                hf_Q = K_val * (abs_Q**(n_exp - 1)) if abs_Q != 0 else 0.0
                
                res_list.append({
                    "루프 (Loop)": l_id,
                    "배관 (Pipe ID)": pid,
                    "방향": "시계(+)" if direction == 1 else "반시계(-)",
                    "최종 유량 (Q) [m³/s]": round(Q_final_loop, 5),
                    "레이놀즈수 (Re)": f"{Re_val:.1f}" if Re_val > 0 else "-",
                    "마찰계수 (f)": f"{f_val:.5f}" if n_exp == 2.0 else f"C={C}",
                    "압력강하 (hf) [m]": round(hf, 5),
                    "압력강하/유량 (hf/Q) [s/m²]": round(abs(hf_Q), 5)
                })
            
        res_df = pd.DataFrame(res_list)
        
        st.markdown("<br><h3>📊 교과서 5단원 형식 최종 풀이표 (루프별 정리)</h3>", unsafe_allow_html=True)
        st.dataframe(res_df.style.background_gradient(subset=['최종 유량 (Q) [m³/s]', '압력강하 (hf) [m]'], cmap='Blues'), use_container_width=True)
        
        st.markdown("---")
        st.write("🔹 양수(+) 최종 유량은 초기에 가정한 방향과 동일하게 흐름을 의미하며, 음수(-)는 반대 방향으로 흐름을 의미합니다.")
        
    except Exception as e:
        st.error(f"연산 중 오류가 발생했습니다: {e}")
        st.info("입력된 데이터의 형식을 다시 확인해 주세요. (누락된 배관 ID 매칭 등)")
