# =============================================================================
# 배관 압력강하 계산기 (Darcy-Weisbach 방정식 및 고급 기능)
# =============================================================================
# 실행 방법:
#   streamlit run pressure_drop.py
# =============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import CoolProp.CoolProp as CP

# ─────────────────────────────────────────────
# [모듈 1] 계산 로직 함수 모음
# ─────────────────────────────────────────────

# 💧 지원 유체 목록
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

def get_fluid_properties(fluid: str, temp_c: float) -> tuple:
    """
    CoolProp을 이용해 유체의 밀도(ρ), 동적 점성계수(μ), 증기압(P_vapor)을 반환합니다.
    입력된 온도에서 유체가 기화되지 않도록 강제로 액체(Liquid) 물성치를 반환합니다.
    """
    T_K = temp_c + 273.15
    P = 101325.0
    p_vapor = 0.0
    
    if fluid.startswith("INCOMP::"):
        # 비압축성 부동액 등은 항상 액체이므로 P 베이스로 정상 계산
        rho = CP.PropsSI("D", "T", T_K, "P", P, fluid)
        mu  = CP.PropsSI("V", "T", T_K, "P", P, fluid)
        p_vapor = 2000.0 # 부동액 증기압 임의 가정치 (거의 증발하지 않음)
    else:
        try:
            # 순수 물질(Water 등)은 포화 액체 건도(Q=0, Saturated Liquid) 라인을 항상 보장하도록 강제
            rho = CP.PropsSI("D", "T", T_K, "Q", 0, fluid)
            mu  = CP.PropsSI("V", "T", T_K, "Q", 0, fluid)
            p_vapor = CP.PropsSI("P", "T", T_K, "Q", 0, fluid) # 해당 온도의 포화 증기압
        except ValueError:
            # 임계 온도를 넘어가 Q=0이 불가능한 극한의 경우 1atm 베이스 폴백
            rho = CP.PropsSI("D", "T", T_K, "P", P, fluid)
            mu  = CP.PropsSI("V", "T", T_K, "P", P, fluid)
            p_vapor = P
            
    return rho, mu, p_vapor

# 배관 재질별 절대 조도 ε (단위: m)
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

# 재질별 기계적 물성치 (E: 탄성계수[Pa], alpha: 열팽창계수[1/C], Sy: 항복강도[Pa])
MECHANICAL_PROPS = {
    "Smooth Pipe (초매끈한 관, ε=0)": {"E": 3e9, "alpha": 5e-5, "Sy": 45e6},
    "PVC (일반 플라스틱 관)": {"E": 3e9, "alpha": 5e-5, "Sy": 45e6},
    "Commercial Steel (상업용 강관)": {"E": 200e9, "alpha": 1.17e-5, "Sy": 250e6},
    "Galvanized Steel (아연도금 강관)": {"E": 200e9, "alpha": 1.17e-5, "Sy": 250e6},
    "Cast Iron (주철관)": {"E": 100e9, "alpha": 1.04e-5, "Sy": 200e6},
    "Concrete (콘크리트관)": {"E": 25e9, "alpha": 1.0e-5, "Sy": 5e6}, 
    "Drawn Tubing (인발 튜브)": {"E": 100e9, "alpha": 1.5e-5, "Sy": 150e6},
    "Stainless Steel (스테인리스 강관)": {"E": 193e9, "alpha": 1.6e-5, "Sy": 205e6},
}

# 관부속품 K-factor (피팅류 계수)
FITTING_LOSSES = {
    "90도 엘보우 (Standard)": 0.75,
    "45도 엘보우 (Standard)": 0.40,
    "티 (Straight run)": 0.60,
    "티 (Branch flow)": 1.80,
}
# 밸브류 K-factor (밸브 저항 계수)
VALVE_LOSSES = {
    "게이트 밸브 (Fully open)": 0.15,
    "글로브 밸브 (Fully open)": 10.0,
    "스윙 체크 밸브": 2.0,
}

# 다양한 한국/일본 배관 표준 규격(KS, JIS) 데이터
PIPE_STANDARDS = {
    "KS D 3576 (스테인리스 강관)": {
        "schedules": ["SCH 5S", "SCH 10S", "SCH 20S", "SCH 40", "SCH 80"],
        "data": {
            "15A (1/2B)":   {"OD": 21.7, "SCH 5S": 1.65, "SCH 10S": 2.1, "SCH 20S": 2.5, "SCH 40": 2.8, "SCH 80": 3.7},
            "20A (3/4B)":   {"OD": 27.2, "SCH 5S": 1.65, "SCH 10S": 2.1, "SCH 20S": 2.5, "SCH 40": 2.9, "SCH 80": 3.9},
            "25A (1B)":     {"OD": 34.0, "SCH 5S": 1.65, "SCH 10S": 2.8, "SCH 20S": 3.0, "SCH 40": 3.4, "SCH 80": 4.5},
            "32A (1 1/4B)": {"OD": 42.7, "SCH 5S": 1.65, "SCH 10S": 2.8, "SCH 20S": 3.0, "SCH 40": 3.6, "SCH 80": 4.9},
            "40A (1 1/2B)": {"OD": 48.6, "SCH 5S": 1.65, "SCH 10S": 2.8, "SCH 20S": 3.0, "SCH 40": 3.7, "SCH 80": 5.1},
            "50A (2B)":     {"OD": 60.5, "SCH 5S": 1.65, "SCH 10S": 2.8, "SCH 20S": 3.5, "SCH 40": 3.9, "SCH 80": 5.5},
            "65A (2 1/2B)": {"OD": 76.3, "SCH 5S": 2.1,  "SCH 10S": 3.0, "SCH 20S": 3.5, "SCH 40": 5.2, "SCH 80": 7.0},
            "80A (3B)":     {"OD": 89.1, "SCH 5S": 2.1,  "SCH 10S": 3.0, "SCH 20S": 4.0, "SCH 40": 5.5, "SCH 80": 7.6},
            "90A (3 1/2B)": {"OD": 101.6,"SCH 5S": 2.1,  "SCH 10S": 3.0, "SCH 20S": 4.0, "SCH 40": 5.7, "SCH 80": 8.1},
            "100A (4B)":    {"OD": 114.3,"SCH 5S": 2.1,  "SCH 10S": 3.0, "SCH 20S": 4.0, "SCH 40": 6.0, "SCH 80": 8.6},
            "125A (5B)":    {"OD": 139.8,"SCH 5S": 2.8,  "SCH 10S": 3.4, "SCH 20S": 5.0, "SCH 40": 6.6, "SCH 80": 9.5},
            "150A (6B)":    {"OD": 165.2,"SCH 5S": 2.8,  "SCH 10S": 3.4, "SCH 20S": 5.0, "SCH 40": 7.1, "SCH 80": 11.0},
            "200A (8B)":    {"OD": 216.3,"SCH 5S": 2.8,  "SCH 10S": 4.0, "SCH 20S": 6.5, "SCH 40": 8.2, "SCH 80": 12.7},
            "250A (10B)":   {"OD": 267.4,"SCH 5S": 3.4,  "SCH 10S": 4.0, "SCH 20S": 6.5, "SCH 40": 9.3, "SCH 80": 15.1},
            "300A (12B)":   {"OD": 318.5,"SCH 5S": 4.0,  "SCH 10S": 4.5, "SCH 20S": 6.5, "SCH 40": 10.3,"SCH 80": 17.4},
        }
    },
    "KS D 3507 (일반 배관용 탄소강관)": {
        "schedules": ["일반배관 (SPP)"],
        "data": {
            "15A (1/2B)":   {"OD": 21.7, "일반배관 (SPP)": 2.8},
            "20A (3/4B)":   {"OD": 27.2, "일반배관 (SPP)": 2.8},
            "25A (1B)":     {"OD": 34.0, "일반배관 (SPP)": 3.2},
            "32A (1 1/4B)": {"OD": 42.7, "일반배관 (SPP)": 3.5},
            "40A (1 1/2B)": {"OD": 48.6, "일반배관 (SPP)": 3.5},
            "50A (2B)":     {"OD": 60.5, "일반배관 (SPP)": 3.8},
            "65A (2 1/2B)": {"OD": 76.3, "일반배관 (SPP)": 4.2},
            "80A (3B)":     {"OD": 89.1, "일반배관 (SPP)": 4.2},
            "100A (4B)":    {"OD": 114.3,"일반배관 (SPP)": 4.5},
            "125A (5B)":    {"OD": 139.8,"일반배관 (SPP)": 4.5},
            "150A (6B)":    {"OD": 165.2,"일반배관 (SPP)": 5.0},
            "200A (8B)":    {"OD": 216.3,"일반배관 (SPP)": 5.8},
            "250A (10B)":   {"OD": 267.4,"일반배관 (SPP)": 6.6},
            "300A (12B)":   {"OD": 318.5,"일반배관 (SPP)": 6.9},
        }
    },
    "KS D 3562 (압력 배관용 탄소강관)": {
        "schedules": ["SCH 40", "SCH 80", "SCH 160"],
        "data": {
            "15A (1/2B)":   {"OD": 21.7, "SCH 40": 2.8, "SCH 80": 3.7, "SCH 160": 4.7},
            "20A (3/4B)":   {"OD": 27.2, "SCH 40": 2.9, "SCH 80": 3.9, "SCH 160": 5.5},
            "25A (1B)":     {"OD": 34.0, "SCH 40": 3.4, "SCH 80": 4.5, "SCH 160": 6.4},
            "32A (1 1/4B)": {"OD": 42.7, "SCH 40": 3.6, "SCH 80": 4.9, "SCH 160": 6.4},
            "40A (1 1/2B)": {"OD": 48.6, "SCH 40": 3.7, "SCH 80": 5.1, "SCH 160": 7.1},
            "50A (2B)":     {"OD": 60.5, "SCH 40": 3.9, "SCH 80": 5.5, "SCH 160": 8.7},
            "65A (2 1/2B)": {"OD": 76.3, "SCH 40": 5.2, "SCH 80": 7.0, "SCH 160": 9.5},
            "80A (3B)":     {"OD": 89.1, "SCH 40": 5.5, "SCH 80": 7.6, "SCH 160": 11.1},
            "100A (4B)":    {"OD": 114.3,"SCH 40": 6.0, "SCH 80": 8.6, "SCH 160": 13.5},
            "125A (5B)":    {"OD": 139.8,"SCH 40": 6.6, "SCH 80": 9.5, "SCH 160": 15.9},
            "150A (6B)":    {"OD": 165.2,"SCH 40": 7.1, "SCH 80": 11.0,"SCH 160": 18.2},
            "200A (8B)":    {"OD": 216.3,"SCH 40": 8.2, "SCH 80": 12.7,"SCH 160": 23.0},
            "250A (10B)":   {"OD": 267.4,"SCH 40": 9.3, "SCH 80": 15.1,"SCH 160": 28.6},
            "300A (12B)":   {"OD": 318.5,"SCH 40": 10.3,"SCH 80": 17.4,"SCH 160": 33.3},
        }
    }
}

def calc_velocity(Q_m3s: float, D: float) -> float:
    """유속(v) 계산: Q / A"""
    # 내경이 0인 경우 방어를 위해 아주 작은 값 추가
    D = max(D, 1e-9)
    A = np.pi / 4.0 * D**2
    return Q_m3s / A

def calc_reynolds(rho: float, v: float, D: float, mu: float) -> float:
    """레이놀즈수(Re) 계산: ρ·v·D / μ"""
    if mu <= 0: return float('inf')
    return (rho * v * D) / mu

def calc_friction_factor(Re: float, D: float, epsilon: float) -> tuple:
    """
    마찰계수(f) 계산:
    - 층류 (Re < 2300): 64/Re
    - 난류 (Re >= 2300): Swamee-Jain 방정식 (1976) 적용
    """
    if Re < 1e-6:
        return 0.0, "정지 (No Flow)"
    if Re < 2300:
        f = 64.0 / Re
        regime = "층류 (Laminar)"
    else:
        D = max(D, 1e-9)
        relative_roughness = epsilon / D
        # Swamee-Jain 방정식
        denom = np.log10(relative_roughness / 3.7 + 5.74 / (Re**0.9))
        f = 0.25 / denom**2
        regime = "전이구간 (Transitional)" if Re < 4000 else "난류 (Turbulent)"
    return f, regime

def calc_pressure_dp(f: float, L: float, D: float, rho: float, v: float, sum_K_fit: float, sum_K_valve: float) -> tuple:
    """
    압력강하 (마찰손실 + 피팅/밸브 손실) 계산 (Pa 단위)
    마찰 손실 (Darcy-Weisbach): ΔP_fric = f * (L/D) * (ρ*v²/2)
    피팅 손실: ΔP_fit = ΣK_fit * (ρ*v²/2)
    밸브 손실: ΔP_valve = ΣK_valve * (ρ*v²/2)
    """
    D = max(D, 1e-9)
    dynamic_pressure = rho * v**2 / 2.0
    dp_fric = f * (L / D) * dynamic_pressure
    dp_fit = sum_K_fit * dynamic_pressure
    dp_valve = sum_K_valve * dynamic_pressure
    dp_total = dp_fric + dp_fit + dp_valve
    return dp_fric, dp_fit, dp_valve, dp_total

def calc_pump_power(dp_pa: float, Q_m3s: float, eff: float) -> float:
    """
    펌프 소요 동력(kW) 계산
    W = (ΔP * Q) / η
    """
    if eff <= 0: return 0.0
    power_watts = (dp_pa * Q_m3s) / (eff / 100.0)
    return power_watts / 1000.0

def get_standard_motor(kw_req: float) -> tuple:
    """
    KS C 4202 기준 3상 유도전동기 (표준 모터) 용량 및 가상의 브랜드 모델명을 반환.
    안전성을 위해 15%의 여유율(Design Margin)을 적용하여 한 단계 높은 표준 용량을 선정.
    """
    # KS 표준 모터 용량 (kW)
    std_sizes = [0.4, 0.75, 1.5, 2.2, 3.7, 5.5, 7.5, 11.0, 15.0, 18.5, 22.0, 30.0, 37.0, 45.0, 55.0, 75.0, 90.0, 110.0, 132.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0]
    
    # 요구 동력의 15% 여유율 적용
    design_kw = kw_req * 1.15
    for size in std_sizes:
        if size >= design_kw:
            return size, f"효성 프리미엄 고효율 전동기(IE3) / {size}kW 급"
    
    return design_kw, f"초대형 맞춤 제작 전동기 / {design_kw:.1f}kW 급"


def solve_parallel_2pipes(Q_total: float, pipe1: dict, pipe2: dict, rho: float, mu: float, epsilon: float) -> tuple:
    """
    병렬 연결된 2개 배관(pipe1, pipe2)의 유량 분배를 이분법(Bisection)으로 수치해석합니다.
    pipe dict 구조: {"L": 길이, "D": 내경, "K": 저항계수합}
    반환값: (Q1, Q2, dp_parallel)
    """
    def calc_dp(Q, p):
        v = calc_velocity(Q, p["D"])
        Re = calc_reynolds(rho, v, p["D"], mu)
        f, _ = calc_friction_factor(Re, p["D"], epsilon)
        _, _, _, dp_total = calc_pressure_dp(f, p["L"], p["D"], rho, v, p["K"], 0)
        return dp_total

    # Bisection을 통해 x (Q1의 비율, 0~1) 탐색
    low, high = 0.0001, 0.9999
    mid = 0.5
    for _ in range(50):
        mid = (low + high) / 2.0
        Q1_test = mid * Q_total
        Q2_test = (1.0 - mid) * Q_total
        
        dp1 = calc_dp(Q1_test, pipe1)
        dp2 = calc_dp(Q2_test, pipe2)
        
        diff = dp1 - dp2
        if abs(diff) < 1e-2: # 0.01 Pa 오차 이내면 종료
            break
            
        # dp1이 더 크면, pipe1의 저항이 크므로 Q1을 줄여야 함 -> x 비율을 낮춘다
        if diff > 0:
            high = mid
        else:
            low = mid
            
    Q1_final = mid * Q_total
    Q2_final = (1.0 - mid) * Q_total
    dp_final = calc_dp(Q1_final, pipe1)
    
    return Q1_final, Q2_final, dp_final

# ─────────────────────────────────────────────
# [모듈 2] Streamlit UI 구성
# ─────────────────────────────────────────────

def build_ui():
    st.set_page_config(page_title="배관 설계 시뮬레이터", page_icon="🚀", layout="wide")
    
    # ── 커스텀 CSS (색상·다이나믹 디자인) ─────────
    st.markdown("""
    <style>
        .stApp { background-color: #F8F9FA; }
        .main-header {
            background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
            padding: 2.5rem; border-radius: 16px;
            color: white; margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        }
        .main-header h1 { margin: 0; font-size: 2.5rem; color: white; font-weight: 800;}
        .main-header p  { margin-top: 0.5rem; opacity: 0.9; font-size: 1.1rem; }
        .section-header { 
            font-size: 1.25rem; font-weight: 700; color: #1E3A8A; 
            margin-top: 1.5rem; margin-bottom: 0.8rem; 
            border-bottom: 2px solid #E5E7EB; padding-bottom: 0.3rem;
        }
        .res-card { 
            background: white; padding: 1.5rem; border-radius: 12px; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 1.5rem;
            transition: transform 0.2s ease-in-out;
        }
        .res-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 15px rgba(0,0,0,0.1);
        }
        .badge { 
            display: inline-flex; align-items: center; justify-content: center;
            padding: 0.35rem 1rem; border-radius: 999px; 
            font-weight: 700; font-size: 1rem; margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .badge-laminar { background:#D1FAE5; color:#065F46; border: 1px solid #34D399; }
        .badge-transitional { background:#FEF3C7; color:#92400E; border: 1px solid #FBBF24; }
        .badge-turbulent { background:#DBEAFE; color:#1E3A8A; border: 1px solid #60A5FA; }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="main-header">
        <h1>🚀 프리미엄 배관 시스템 설계 계산기</h1>
        <p>Darcy-Weisbach 직관 손실 &nbsp;|&nbsp; 피팅 및 밸브 손실 계산 &nbsp;|&nbsp; 필요 펌프 동력</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ══════════════════════════════════════════
    # 사이드바 : 입력 패널
    # ══════════════════════════════════════════
    with st.sidebar:
        st.header("⚙️ 시스템 조건 입력")
        
        # ── 1. 유체 정보 ───────────────────────
        st.subheader("💧 1. 유체(Fluid) 특성")
        
        fluid_options_list = list(FLUID_OPTIONS.values()) + ["직접 입력 (Manual)"]
        fluid_display = st.selectbox("유체 종류", fluid_options_list, index=0)
        
        if fluid_display == "직접 입력 (Manual)":
            rho = st.number_input("밀도 ρ (kg/m³)", value=1000.0, step=10.0, format="%.2f")
            mu = st.number_input("동적 점성계수 μ (Pa·s)", value=0.001, step=0.0001, format="%.6f")
            p_vapor = st.number_input("유체 증기압 (Pa)", value=2300.0, step=100.0, help="NPSH 계산에 사용됩니다. (20도 물 기준 약 2300 Pa)")
        else:
            fluid_key = [k for k, v in FLUID_OPTIONS.items() if v == fluid_display][0]
            temp_c = st.number_input("유체 온도 (°C)", min_value=-50.0, max_value=500.0, value=20.0, step=1.0)
            
            try:
                rho, mu, p_vapor = get_fluid_properties(fluid_key, temp_c)
                st.info(f"**밀도:** {rho:.2f} kg/m³  \n**점성계수:** {mu:.4e} Pa·s  \n**증기압:** {p_vapor:.0f} Pa")
            except Exception as e:
                st.error(f"⚠️ 물성치 계산 오류: {e}")
                rho, mu, p_vapor = 1000.0, 0.001, 2300.0

        st.markdown("---")
        # ── 2. 배관 제원 ───────────────────────
        st.subheader("🔩 2. 배관(Pipe) 및 유량 제원")
        
        q_unit = st.radio("유량 단위 선택", ["m³/s", "L/min"], horizontal=True)
        if q_unit == "L/min":
            Q_input = st.number_input("설계 체적 유량 (L/min)", min_value=0.1, value=100.0, step=10.0)
            Q_m3s = Q_input / 60000.0
        else:
            Q_input = st.number_input("설계 체적 유량 (m³/s)", min_value=0.0001, value=0.005, step=0.001, format="%.4f")
            Q_m3s = Q_input

        d_input_method = st.radio("배관 내경 입력 방식", ["직접 입력", "규격표에서 선택 (KS/JIS 배관)"], horizontal=True)
        
        if d_input_method == "직접 입력":
            d_unit = st.radio("내경 단위 선택", ["mm", "m"], horizontal=True)
            if d_unit == "mm":
                D_input = st.number_input("배관 내경 (mm)", min_value=1.0, value=50.0, step=5.0)
                D = D_input / 1000.0
            else:
                D_input = st.number_input("배관 내경 (m)", min_value=0.001, value=0.05, step=0.01)
                D = D_input
            pipe_std_name = "직접 입력"
        else:
            pipe_std_name = st.selectbox("배관 표준 규격", list(PIPE_STANDARDS.keys()))
            selected_std = PIPE_STANDARDS[pipe_std_name]
            
            col_sc1, col_sc2 = st.columns(2)
            with col_sc1:
                # 50A 정도가 기본값으로 오게 설정 (없으면 0)
                default_idx = 5 if len(selected_std["data"]) > 5 else 0
                nps_select = st.selectbox("호칭 지름", list(selected_std["data"].keys()), index=default_idx)
            with col_sc2:
                # SCH 40이 있으면 기본값, 없으면 0
                sch_options = selected_std["schedules"]
                sch_idx = sch_options.index("SCH 40") if "SCH 40" in sch_options else 0
                sch_select = st.selectbox("호칭 두께 (Schedule)", sch_options, index=sch_idx)
                
            pipe_od = selected_std["data"][nps_select]["OD"]
            # 선택한 스케줄이 해당 관경에 없을 경우 방어 코드
            if sch_select in selected_std["data"][nps_select]:
                pipe_t = selected_std["data"][nps_select][sch_select]
            else:
                st.warning(f"⚠️ 선택하신 관경({nps_select})에 {sch_select} 규격이 존재하지 않아 가장 두꺼운 두께로 대체합니다.")
                # OD 제외 가장 큰 값 찾기
                t_candidates = [v for k, v in selected_std["data"][nps_select].items() if k != "OD"]
                pipe_t = max(t_candidates) if t_candidates else 0.0
                
            selected_inner_mm = pipe_od - (2 * pipe_t)
            
            st.info(f"📏 외경: **{pipe_od}** mm / 두께: **{pipe_t}** mm\n\n👉 계산된 실제 내경: **{selected_inner_mm:.2f} mm**")
            D = selected_inner_mm / 1000.0
            
        L = st.number_input("배관 총 직관 길이 (m)", min_value=0.1, value=50.0, step=5.0)
        
        material = st.selectbox("배관 재질 (절대조도)", list(ROUGHNESS.keys()))
        epsilon = ROUGHNESS[material]
        
        st.markdown("---")
        # ── 3. 부속품 및 밸브 손실 ───────────────────────
        st.subheader("🌟 3. 부속품 및 밸브 손실")
        st.caption("배관 시스템 내 설치된 밸브 및 피팅류 갯수를 입력하세요.")
        
        col_f, col_v = st.columns(2)
        fitting_counts, valve_counts = {}, {}
        with col_f:
            st.markdown("**[ 피팅류 ]**")
            for item in FITTING_LOSSES.keys():
                fitting_counts[item] = st.number_input(f"{item}", min_value=0, value=0, step=1, key=f"f_{item}")
        with col_v:
            st.markdown("**[ 밸브류 ]**")
            for item in VALVE_LOSSES.keys():
                valve_counts[item] = st.number_input(f"{item}", min_value=0, value=0, step=1, key=f"v_{item}")
            
        sum_K_fit = sum(count * FITTING_LOSSES[item] for item, count in fitting_counts.items())
        sum_K_valve = sum(count * VALVE_LOSSES[item] for item, count in valve_counts.items())
        if (sum_K_fit + sum_K_valve) > 0:
            st.success(f"**피팅 저항 (ΣK) = {sum_K_fit:.2f} / 밸브 저항 (ΣK) = {sum_K_valve:.2f}**")

        st.markdown("---")
        # ── 4. 펌프 동력 및 NPSH ───────────────────────
        st.subheader("⚡ 4. 펌프 동력 및 NPSH")
        pump_eff = st.slider("펌프 모터 종합 효율 (%)", min_value=30, max_value=100, value=70, step=1)
        
        with st.expander("🌀 NPSH (가용 흡입수두) 및 캐비테이션 검토 설정", expanded=False):
            st.markdown("펌프의 캐비테이션을 방지하기 위해 가용 흡입수두(NPSHa)를 진단합니다.")
            npsh_p_abs = st.number_input("흡입 탱크 표면 절대압력 (bar)", value=1.013, step=0.1, help="오픈된 탱크라면 대기압(1.013 bar)을 사용합니다.")
            npsh_static = st.number_input("펌프 중심축 기준 흡입 수위 (m)", value=2.0, step=0.5, help="탱크 수위가 펌프보다 높으면 양수(+), 낮으면 음수(-)를 입력하세요. (흡입양정)")
            npsh_l = st.number_input("흡입측 배관 직관 길이 (m)", min_value=0.1, value=5.0, step=1.0, help="위에서 입력한 '총 배관 길이' 중 흡입측에 해당하는 길이")
            npsh_k = st.number_input("흡입측 밸브/피팅 저항 (K)", min_value=0.0, value=1.5, step=0.5, help="스트레이너, 풋밸브 등 흡입측의 총 K 팩터")
            
        st.markdown("<br>", unsafe_allow_html=True)
        calc_btn = st.button("🚀 전체 시스템 시뮬레이션 결과 보기", type="primary", use_container_width=True)

    if 'calc_run' not in st.session_state:
        st.session_state['calc_run'] = False
        
    if calc_btn:
        st.session_state['calc_run'] = True

    # ══════════════════════════════════════════
    # 메인 화면 : 결과 출력
    # ══════════════════════════════════════════
    if st.session_state['calc_run']:
        # 단일 설계점 포인트 계산 실행
        v = calc_velocity(Q_m3s, D)
        Re = calc_reynolds(rho, v, D, mu)
        f, regime = calc_friction_factor(Re, D, epsilon)
        dp_fric, dp_fit, dp_valve, dp_total = calc_pressure_dp(f, L, D, rho, v, sum_K_fit, sum_K_valve)
        pow_kw = calc_pump_power(dp_total, Q_m3s, pump_eff)
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 기본 시뮬레이션 결과", "⚖️ 단일 조건 비교 분석", "💸 경제적 최적 관경(Economic Diameter)", "🛡️ 구조 및 환경 안전성 진단", "🔀 다중 관로(직렬/병렬) 분석"])
        
        with tab1:
            # ── 흐름 상태 뱃지 출력 ───────────────
            if "층류" in regime: 
                regime_class = "badge-laminar"
                icon = "🟢"
            elif "전이" in regime: 
                regime_class = "badge-transitional"
                icon = "🟠"
            else: 
                regime_class = "badge-turbulent"
                icon = "🔵"
            
            st.markdown(
                f"<div><span class='badge {regime_class}'>{icon} 유체 흐름 체제: <b>{regime}</b></span></div>", 
                unsafe_allow_html=True
            )
            
            if "전이" in regime:
                st.warning("⚠️ **경고:** 현재 결괏값은 **천이 영역(Transitional Flow, 2300 ≤ Re < 4000)**에 속합니다. 이 영역에서는 점성력과 관성력의 상호작용으로 인해 유동이 매우 불안정하며 마찰계수(f)의 오차가 가장 큽니다. **반드시 배관경이나 설정 유량을 조절하여 안정적인 층류나 난류 영역으로 설계를 변경하세요.**")
            
            col1, col2 = st.columns([1.6, 1])
            with col1:
                st.markdown("<div class='res-card'>", unsafe_allow_html=True)
                st.markdown("<div class='section-header'>📊 주요 성능 지표 (KPI)</div>", unsafe_allow_html=True)
                
                # NPSHa (가용 흡입수두) 연산
                p_abs_pa = npsh_p_abs * 1e5
                # 흡입측 마찰/피팅 손실 (Pa) -> 수두(m)로 변환
                npsh_dp_fric, npsh_dp_fit, _, npsh_dp_total = calc_pressure_dp(f, npsh_l, D, rho, v, npsh_k, 0.0)
                h_loss_suction = npsh_dp_total / (rho * 9.81)
                
                npsha = ((p_abs_pa - p_vapor) / (rho * 9.81)) + npsh_static - h_loss_suction
                
                # 최종 결과 매트릭 출력
                c1, c2, c3 = st.columns(3)
                c1.metric("총 시스템 압력강하", f"{dp_total / 1e5:.4f} bar")
                c2.metric("총 시스템 압력강하", f"{dp_total:,.1f} Pa")
                c3.metric("필요 펌프 동력 (전력)", f"{pow_kw:.3f} kW", help=f"시스템 효율 {pump_eff}% 기준 소요 동력")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # 중간 계산 결과 출력
                c4, c5, c6 = st.columns(3)
                c4.metric("평균 유속 (v)", f"{v:.3f} m/s")
                c5.metric("가용 흡입수두 (NPSHa)", f"{npsha:.2f} m", help="펌프 캐비테이션 방지를 위해 펌프의 NPSHr 보다 이 값이 최소 0.5~1m 이상 커야 합니다.")
                c6.metric("최종 마찰계수 (f)", f"{f:.5f}")
                
                if npsha < 2.5:
                    st.error(f"🚨 **캐비테이션 경고:** NPSHa 값이 너무 낮습니다 ({npsha:.2f}m). 흡입측 관경을 늘리거나 펌프 수위를 낮춰 흡입조건을 개선하세요.")
                elif npsha < 5.0:
                    st.warning(f"⚠️ **주의 요망:** NPSHa 값이 {npsha:.2f}m 입니다. 선정하려는 펌프의 NPSHr을 반드시 확인하세요.")
                    
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col2:
                st.markdown("<div class='res-card'>", unsafe_allow_html=True)
                st.markdown("<div class='section-header'>분석 요약 보고서</div>", unsafe_allow_html=True)
                
                # 0으로 나누기 방지
                safe_total = dp_total if dp_total > 0 else 1e-9
                
                st.write(f"🔹 **직관 마찰 손실**: {dp_fric/1e5:.4f} bar ({dp_fric/safe_total*100:.1f}%)")
                st.write(f"🔸 **피팅류 손실**: {dp_fit/1e5:.4f} bar ({dp_fit/safe_total*100:.1f}%)")
                st.write(f"🔺 **밸브류 손실**: {dp_valve/1e5:.4f} bar ({dp_valve/safe_total*100:.1f}%)")
                st.write(f"📏 **단위 길이당 직관 압력구배**: {dp_fric/L:.1f} Pa/m")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                st.progress(min(dp_fric/safe_total, 1.0), text="🟦 직관 마찰 손실 (Friction Loss)")
                st.progress(min(dp_fit/safe_total, 1.0), text="🟧 피팅류 손실 (Fitting Loss)")
                st.progress(min(dp_valve/safe_total, 1.0), text="🟥 밸브류 손실 (Valve Loss)")
                st.markdown("</div>", unsafe_allow_html=True)

        with tab2:
            st.markdown("<div class='res-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-header'>⚖️ 대안 설계 (Alternative Design) 비교</div>", unsafe_allow_html=True)
            st.info("좌측 사이드바의 기본 조건(Base Case)을 유지한 채, **아래에서 선택한 변수 1개만 변경** 시 결과를 비교합니다.")
            
            compare_var = st.selectbox("변경할 변수 선택", [
                "1. 유체 온도 변경 (물성치 변화 적용)",
                "2. 체적 유량 변경",
                "2. 배관 내경 강제 변경",
                "2. 배관 총 직관 길이 변경",
                "3. 수동 저항계수(K) 추가",
                "4. 펌프 종합 효율 변경"
            ])
            
            alt_temp_c = temp_c if 'temp_c' in locals() else 20.0
            alt_Q_m3s = Q_m3s
            alt_D = D
            alt_L = L
            alt_K_add = 0.0
            alt_pump_eff = pump_eff
            
            st.markdown("---")
            if "온도" in compare_var:
                if fluid_display == "직접 입력 (Manual)":
                    st.warning("⚠️ 직접 입력하신 유체는 온도 변화에 따른 물성치 자동 계산이 불가합니다.")
                else:
                    alt_temp_c = st.number_input("대안 유체 온도 (°C)", value=alt_temp_c + 10.0, step=1.0)
            elif "유량" in compare_var:
                if q_unit == "L/min":
                    alt_Q_input = st.number_input("대안 체적 유량 (L/min)", value=Q_input + 20.0, step=10.0)
                    alt_Q_m3s = alt_Q_input / 60000.0
                else:
                    alt_Q_input = st.number_input("대안 체적 유량 (m³/s)", value=Q_input + 0.001, step=0.001, format="%.4f")
                    alt_Q_m3s = alt_Q_input
            elif "내경" in compare_var:
                alt_D_input = st.number_input("대안 배관 내경 (mm)", value=(D * 1000.0) + 10.0, step=5.0)
                alt_D = alt_D_input / 1000.0
            elif "길이" in compare_var:
                alt_L = st.number_input("대안 배관 길이 (m)", value=L + 10.0, step=5.0)
            elif "저항계수" in compare_var:
                alt_K_add = st.number_input("추가할 저항계수 K (단위 무차원, 필터류/밸브 추가 가정)", value=5.0, step=1.0)
            elif "효율" in compare_var:
                alt_pump_eff = st.slider("대안 펌프 효율 (%)", min_value=30, max_value=100, value=min(pump_eff + 10, 100), step=1)
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            # 대안 재계산 로직
            c_rho, c_mu = rho, mu
            if "온도" in compare_var and fluid_display != "직접 입력 (Manual)":
                try:
                    c_rho, c_mu, _ = get_fluid_properties(fluid_key, alt_temp_c)
                except:
                    pass
                    
            alt_sum_K_valve = sum_K_valve + alt_K_add
            
            a_v = calc_velocity(alt_Q_m3s, alt_D)
            a_Re = calc_reynolds(c_rho, a_v, alt_D, c_mu)
            a_f, a_regime = calc_friction_factor(a_Re, alt_D, epsilon)
            a_dp_fric, a_dp_fit, a_dp_valve, a_dp_total = calc_pressure_dp(a_f, alt_L, alt_D, c_rho, a_v, sum_K_fit, alt_sum_K_valve)
            a_power_kw = calc_pump_power(a_dp_total, alt_Q_m3s, alt_pump_eff)
            
            # 비교 UI 출력
            col_b, col_a = st.columns(2)
            
            diff_dp = a_dp_total - dp_total
            diff_pow = a_power_kw - pow_kw
            diff_v = a_v - v
            
            with col_b:
                st.markdown("<div style='background-color:#F8FAFC; padding:1.5rem; border-radius:12px; border-top: 4px solid #94A3B8;'>", unsafe_allow_html=True)
                st.markdown("<h3 style='color:#334155; margin-top:0;'>🔵 기본 조건 (Base)</h3>", unsafe_allow_html=True)
                st.write(f"**평균 유속**: {v:.3f} m/s")
                st.write(f"**총 압력강하**: {dp_total/1e5:.4f} bar")
                st.write(f"**필요 펌프 동력**: {pow_kw:.3f} kW")
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col_a:
                st.markdown("<div style='background-color:#EFF6FF; padding:1.5rem; border-radius:12px; border-top: 4px solid #3B82F6; box-shadow:0 4px 6px rgba(0,0,0,0.05);'>", unsafe_allow_html=True)
                st.markdown("<h3 style='color:#1E3A8A; margin-top:0;'>🟠 대안 조건 (Alternative)</h3>", unsafe_allow_html=True)
                st.metric("평균 유속 (m/s)", f"{a_v:.3f}", delta=f"{diff_v:.3f}", delta_color="normal")
                st.metric("총 압력강하 (bar)", f"{a_dp_total/1e5:.4f}", delta=f"{diff_dp/1e5:.4f}", delta_color="inverse")
                st.metric("필요 펌프 동력 (kW)", f"{a_power_kw:.3f}", delta=f"{diff_pow:.3f}", delta_color="inverse")
                st.markdown("</div>", unsafe_allow_html=True)
                
            st.markdown("</div>", unsafe_allow_html=True)

        with tab3:
            st.markdown("<div class='res-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-header'>🇰🇷 한국 기준 현장 맞춤형 파이프 관경 최적화 (KS 규격 맵핑 & LCC 분석)</div>", unsafe_allow_html=True)
            st.markdown("""
            **생애주기비용 (LCC: Life Cycle Cost) 기반 최적화 논리**: 
            과거 단순 경험식이 아닌, 대한민국 건설 현장의 실제 지정 **배관 규격(KS 기준)** 내에서 탐색을 수행합니다.
            뿐만 아니라 현재 산업 트렌드인 **전력요금 인상률(인플레이션)**, **탄소배출권(K-ETS)** 구매 비용, 그리고 **유지보수 비용**까지 전부 연간 등가비용(EUAC, NPV)으로 환산하여 가장 저렴한 '진짜 최적 관경(KS 사이즈)'을 찾아냅니다.
            """, unsafe_allow_html=True)
            
            st.info("💡 파이썬 수치해석 엔진을 통해 KS 규격(스케줄) 테이블에 존재하는 내경 치수들만 선별적으로 시뮬레이션하여 가장 경제적인 호칭경(A)을 도출합니다.")
            
            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                eco_elec = st.number_input("산업용 전기요금 (원/kWh)", value=150.0, step=5.0, help="2026년 한국전력 산업용(을) 중간부하 요금 등 가중평균")
            with col_b2:
                eco_ir = st.number_input("기준/조달 금리 (%)", value=2.50, step=0.1, help="할인율 적용 (예: 2026 한국은행 기준금리)")
            with col_b3:
                eco_years = st.number_input("자본회수 (대출상환) 기간 (년)", value=20, step=1, help="투입된 초기 자본을 몇 년에 걸쳐 회수할지(Payback) 정합니다. 이 기간을 기준으로 자본회수계수(CRF)가 적용되어 이자율과 함께 연간 비용으로 할부 환산됩니다.")

            with st.expander("💎 심화 경제 지표 및 건설 공법 변수 설정", expanded=True):
                st.markdown("**[1] 운전 및 환경 변수 (OPEX & ESG)**")
                col_a1, col_a2, col_a3, col_a4 = st.columns(4)
                with col_a1:
                    eco_inf_e = st.number_input("전기요금 연평균 인상률 (%)", value=3.0, step=0.1, help="물가상승 및 한전 요금 에스컬레이션 반영")
                with col_a2:
                    eco_carbon_price = st.number_input("탄소배출권 (원/tCO2eq)", value=15000, step=1000, help="K-ETS 톤당 배출권 거래 가격")
                with col_a3:
                    eco_maint_rate = st.number_input("연간 유지보수율 (%)", value=2.0, step=0.1, help="매년 초기 투자비의 N%를 설비보전에 사용")
                with col_a4:
                    eco_hours = st.number_input("연간 펌프 가동시간 (hr/yr)", value=8000, step=100)
                    
                st.markdown("---")
                st.markdown("**[2] 기계 장비(Pump) 상세 설정** (조달청 나라장터 MAS 펌프 단가 및 효율 회귀식 자동 적용)")
                col_p1, col_p2, col_p3 = st.columns(3)
                with col_p1:
                    eco_auto_pump = st.checkbox("⚙️ MAS 회귀식 자동 산출 켜기", value=True, help="체크 시 필요 동력(kW)과 유량/양정을 기반으로 조달청 MAS 펌프 회귀식을 적용합니다.")
                with col_p2:
                    manual_pump_price = st.number_input("수동 펌프 장비비 (원)", value=2000000, step=100000, disabled=eco_auto_pump, help="자동 산출이 꺼졌을 때 사용할 펌프 단가")
                with col_p3:
                    manual_pump_eff = st.number_input("수동 펌프 효율 (%)", value=65.0, step=1.0, disabled=eco_auto_pump, help="자동 산출이 꺼졌을 때 사용할 펌프 효율")
                
                st.markdown("---")
                st.markdown("**[3] 배관 건설 및 시공 환경 설정 (교과서 설비 투자비 산출 인자 반영)**")
                col_c1, col_c2, col_c3 = st.columns(3)
                with col_c1:
                    eco_material_k = st.number_input("순수 직관 자재비 계수 (원/m/m)", value=500000, step=50000, help="지름 1m당 미터당 순수 파이프(Straight Pipe) 자재비")
                    eco_fitting_pct = st.number_input("이음쇠류(피팅/밸브) 할증 (%)", value=35.0, step=5.0, help="교과서 기준: 방향 전환 및 밸브 부속품으로 인한 직관 자재비 대비 추가분율 (보통 30~50%)")
                with col_c2:
                    eco_install_type = st.selectbox("배관 시공 공법 (설치 환경)", ["지상 노출 (파이프랙/지지대/트렌치)", "지중 매설 (땅파기/토목굴착 및 되메우기 포함)"], index=0)
                with col_c3:
                    eco_labor_pct = st.number_input("현장 노무비/경비 할증 (%)", value=150.0, step=10.0, help="자재비 대비 현장 용접공/배관공 등 인건비 및 경비 비율")
                    
            if d_input_method == "직접 입력":
                st.warning("현재 '직접 입력' 모드로 설계 중입니다. 경제관경(LCC) 자동 탐색은 KS 규격을 기준으로 동작합니다. 가장 많이 쓰이는 **KS D 3507 (일반 배관용 탄소강관)** 기준으로 강제 시뮬레이션을 진행합니다.")
                opt_std_name = "KS D 3507 (일반 배관용 탄소강관)"
                opt_std_data = PIPE_STANDARDS[opt_std_name]
            else:
                opt_std_name = pipe_std_name
                opt_std_data = PIPE_STANDARDS[pipe_std_name]
                
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                eco_sch = st.selectbox(f"[{opt_std_name}] 적용할 두께(Schedule) 규격", opt_std_data["schedules"], index=0, help="현장 시방서에 지정된 스케줄을 고르세요.")
            with col_s2:
                eco_carbon_factor = st.number_input("전력 탄소배출계수 (tCO2/MWh)", value=0.4594, step=0.0100, format="%.4f", help="환경부 발표 계수")

            st.markdown("<br>", unsafe_allow_html=True)
            
            # --- 재무 계수 (NPV/EUAC) 계산 ---
            ir = eco_ir / 100.0
            inf_e = eco_inf_e / 100.0
            N = eco_years

            # 1. 자본회수계수 (CRF: Capital Recovery Factor)
            crf = (ir * (1 + ir)**N) / ((1 + ir)**N - 1) if ir > 0 else 1.0 / N
            # 2. 표준 현가계수 (유지보수 연금 등가)
            pvf_std = (1.0 - (1.0 + ir)**-N) / ir if ir > 0 else N
            # 3. 물가상승 잔존 현가계수 (전기요금 에스컬레이션 반영)
            if ir == inf_e:
                pvf_energy = N / (1.0 + ir)
            else:
                x = (1.0 + inf_e) / (1.0 + ir)
                pvf_energy = (1.0 / (1.0 + inf_e)) * x * (1.0 - x**N) / (1.0 - x)
                
            # 균등화 연간비용(EUAC = 평준화) 보정계수
            lvl_energy_factor = pvf_energy * crf
            lvl_maint_factor = pvf_std * crf  # 사실상 1.0 이지만 공식 원칙을 따름

            # --- KS 규격 기반 이산형(Discrete) 탐색 ---
            ks_candidates = []
            for nps, sch_data in opt_std_data["data"].items():
                if eco_sch in sch_data:
                    t = sch_data[eco_sch]
                    od = sch_data["OD"]
                    id_m = (od - 2*t) / 1000.0
                    if id_m > 0:
                        ks_candidates.append({"Name": nps, "D_m": id_m})
            
            eco_data = []
            min_cost = float('inf')
            opt_d_name = ""
            opt_d_val = 0.0
            opt_v = 0.0
            
            # 결과 저장용 임시 딕셔너리
            opt_details = {}

            for cand in ks_candidates:
                t_name = cand["Name"]
                test_d = cand["D_m"]
                
                t_v = calc_velocity(Q_m3s, test_d)
                t_Re = calc_reynolds(rho, t_v, test_d, mu)
                t_f, _ = calc_friction_factor(t_Re, test_d, epsilon)
                
                # 경제관경에서도 직관부 + 피팅 등 총합산 압력강하 기준으로 보수적 접근 산정
                t_dp_fric, t_dp_fit, t_dp_valve, t_dp_total = calc_pressure_dp(t_f, L, test_d, rho, t_v, sum_K_fit, sum_K_valve)
                
                # --- 동적 펌프 효율 및 장비비 자동 산정 계통 ---
                if eco_auto_pump:
                    H_m = t_dp_total / (rho * 9.81)
                    q_m3h = Q_m3s * 3600.0
                    # 1. 펌프 효율 자동 회귀식
                    base_eff = 45.0 + (15.0 * np.log10(max(q_m3h, 1.0)))
                    base_eff = min(base_eff, 85.0)
                    head_penalty = max(0.0, (H_m - 30.0) * 0.15)
                    curr_eff = max(base_eff - head_penalty, 30.0)
                else:
                    curr_eff = manual_pump_eff
                    
                t_kw = calc_pump_power(t_dp_total, Q_m3s, curr_eff)
                
                if eco_auto_pump:
                    # 2. 조달청 MAS 등록 펌프 단가 회귀식 적용
                    # 기준: 단가 = 780,000 * (kW)^0.82 + 450,000 (모터 및 기본 부속 포함 추정식)
                    pump_capex = 780000.0 * (t_kw ** 0.82) + 450000.0
                else:
                    pump_capex = manual_pump_price
                
                # 펌프 모터 표준규격 맵핑
                std_motor_kw, motor_model = get_standard_motor(t_kw)
                
                # --- 배관 시공비 세부 연산 (유체시스템 교과서 플랜트 원가 산정 인자) ---
                # 추가로, 사이드바에서 입력한 실제 밸브 및 피팅 갯수를 자재비(CapEx)에 실비용으로 합산
                total_fittings_count = sum(fitting_counts.values()) if 'fitting_counts' in locals() else 0
                total_valves_count = sum(valve_counts.values()) if 'valve_counts' in locals() else 0
                
                # 관경(D)이 커질수록 밸브/피팅 단가가 급증함을 모델링 (예: 밸브 1개당 D*200만원, 파이프피팅 D*30만원)
                real_fitting_cost = (total_fittings_count * test_d * 300000) + (total_valves_count * test_d * 2000000)
                
                t_capex_material = eco_material_k * test_d * L
                # 교과서 비율(직관의 N%) + 실제 입력 밸브/피팅 단가 합산
                t_capex_fittings = (t_capex_material * (eco_fitting_pct / 100.0)) + real_fitting_cost
                t_capex_base = t_capex_material + t_capex_fittings  # 총 자재비
                
                install_factor = (eco_labor_pct / 100.0)
                if "매설" in eco_install_type:
                    install_factor += 0.8  # 지중 매설 시 토목공사(터파기/되메우기) 비용으로 인한 막대한 공사비 가산
                    
                pipe_capex = t_capex_base * (1.0 + install_factor)
                
                # 1. 연간 자본비용 (Capex EUAC: 파이프 자재시공비 + 펌프 기계설비비 합산)
                capex = pipe_capex + pump_capex
                euac_capex = capex * crf
                
                # 2. 연간 전력비용 평준화 (Energy EUAC)
                e1 = t_kw * eco_hours * eco_elec
                euac_energy = e1 * lvl_energy_factor
                
                # 3. 연간 탄소부과금 평준화 (Carbon EUAC) - 전기세 인상률과 동조한다고 가정
                carbon1 = (t_kw * eco_hours / 1000.0) * eco_carbon_factor * eco_carbon_price
                euac_carbon = carbon1 * lvl_energy_factor
                
                # 4. 연간 기계/설비 유지보수비 (O&M EUAC)
                maint1 = capex * (eco_maint_rate / 100.0)
                euac_maint = maint1 * lvl_maint_factor
                
                total_annual = euac_capex + euac_energy + euac_carbon + euac_maint
                
                # 레이블명 축소 (그래프 시인성을 위해 A만 표기)
                display_name = t_name.split(' (')[0]
                
                eco_data.append({
                    "규격 (호칭경)": display_name,
                    "실내경_m": test_d,
                    "💡 운전 전력비용 (Levelized)": euac_energy,
                    "🌿 탄소배출 패널티 (Levelized)": euac_carbon,
                    "🔧 유지보수 O&M (Levelized)": euac_maint,
                    "🏗️ 자본상각 (Capex)": euac_capex,
                    "총 연간비용": total_annual
                })
                
                if total_annual < min_cost:
                    min_cost = total_annual
                    opt_d_name = t_name
                    opt_d_val = test_d
                    opt_v = t_v
                    opt_details = {
                        "e": euac_energy,
                        "c": euac_carbon,
                        "m": euac_maint,
                        "k": euac_capex,
                        "eff": curr_eff,
                        "kw": t_kw,
                        "std_kw": std_motor_kw,
                        "motor_model": motor_model,
                        "pump_cost": pump_capex,
                        "extra_fitting_cost": real_fitting_cost
                    }
                    
            df_eco = pd.DataFrame(eco_data)
            df_eco.set_index("규격 (호칭경)", inplace=True)
            
            # 결과 시각화
            res_col1, res_col2 = st.columns([1, 1.3])
            with res_col1:
                st.markdown("<div style='background-color:#F0FDF4; padding:1.5rem; border-radius:12px; border: 2px solid #22C55E; box-shadow:0 4px 6px rgba(0,0,0,0.05);'>", unsafe_allow_html=True)
                st.markdown("<h3 style='color:#166534; margin-top:0;'>✨ 타겟 KS 최적 관경</h3>", unsafe_allow_html=True)
                st.markdown(f"<h2 style='color:#064E3B; margin:-10px 0 10px 0;'>👉 {opt_d_name} ({eco_sch})</h2>", unsafe_allow_html=True)
                st.write(f"최적 내경 ($D_{{opt}}$) : **{opt_d_val*1000:.1f} mm**")
                st.write(f"경제 유속 ($V_{{opt}}$) : **{opt_v:.2f} m/s**")
                
                if eco_auto_pump:
                    st.caption(f"↳ 필요 동력: **{opt_details['kw']:.2f} kW** (펌프 효율 **{opt_details['eff']:.1f}%**, 펌프 가격: **{opt_details['pump_cost']:,.0f}원**)")
                    st.success(f"↳ **적용 모터:** {opt_details['motor_model']} (설계 여유율 15% 반영)")
                    
                if opt_details.get("extra_fitting_cost", 0) > 0:
                    st.info(f"💡 추가된 밸브 및 피팅 실물 자재비 (+ **{opt_details['extra_fitting_cost']:,.0f} 원**) 이 전체 초기 자본(Capex)에 반영되었습니다.")
                
                st.markdown("---")
                st.write(f"가장 저렴한 총 연간비용: **{min_cost:,.0f} 원/년**")
                
                st.caption(f"▪전기료: {opt_details['e']:,.0f}원 ▪탄소: {opt_details['c']:,.0f}원")
                st.caption(f"▪O&M: {opt_details['m']:,.0f}원 ▪자본시공: {opt_details['k']:,.0f}원")
                st.markdown("</div>", unsafe_allow_html=True)
                
                if D < opt_d_val * 0.8:
                    st.error("🚨 **치명적 오류**: 현재 배관경이 너무 작습니다. 전력세와 탄소세가 폭증하여 LCC(생애주기비용) 관점에서 막대한 적자가 예상됩니다.")
                elif D > opt_d_val * 1.2:
                    st.warning("⚠️ **과투자 주의**: 관경이 커서 운전비는 적지만 초기 설비 원가 및 거대 밸브 등 자재비 낭비가 큽니다.")
                else:
                    st.success("✅ **우수 평가**: 현재 입력하신 관경(D)이 복합 LCC 진단하에 산출된 KS 경제 최적 스펙과 거의 일치합니다. 대단합니다!")
                    
            with res_col2:
                # 차트 그리기
                st.markdown("**📉 LCC (Life Cycle Cost) 항목별 KS규격 비용 동향**")
                st.bar_chart(
                    df_eco[["💡 운전 전력비용 (Levelized)", "🌿 탄소배출 패널티 (Levelized)", "🔧 유지보수 O&M (Levelized)", "🏗️ 자본상각 (Capex)"]],
                    height=350,
                    use_container_width=True
                )
            
            st.markdown("</div>", unsafe_allow_html=True)
                
        with tab4:
            st.markdown("<div class='res-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-header'>🚧 극한 환경 및 내압 파열 안전성 분석 (한국 기후 반영)</div>", unsafe_allow_html=True)
            st.info("💡 한반도의 뚜렷한 4계절(폭염 및 혹한)로 인한 배관의 열팽창/열수축과, 펌프 가동 시 배관이 대기압이나 수압에 의해 터지지 않는지 파열(Bursting) 안전성을 복합적으로 진단합니다.")

            # --- 사용자 입력 ---
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.markdown("**1. 배관 설치 및 환경 온도**")
                install_temp = st.number_input("배관 시공 시 온도 (°C)", value=15.0, step=1.0)
                max_env_temp = st.number_input("여름철 직사광 최고 온도 (°C)", value=40.0, step=1.0, help="한반도 폭염 및 복사열 기준 극단치")
                min_env_temp = st.number_input("겨울철 최저 온도 (°C)", value=-20.0, step=1.0, help="한반도 혹한 기준 극단치")
            with col_t2:
                st.markdown("**2. 내압 파열 및 외력 안전 여유**")
                safety_factor = st.number_input("목표 내압 안전율 (Safety Factor)", value=3.0, step=0.5, help="항복강도 대비 여유. 일반 산업용 배관은 3.0 ~ 4.0 이상 권장")
                surge_multiplier = st.number_input("수격작용(Water Hammer) 할증 계수", value=1.5, step=0.1, help="펌프 급기동/정지 및 밸브 급개폐 시 순간적으로 튀어오르는 압력 스파이크 반영 비율")

            # 기계적 물성치 추출
            props = MECHANICAL_PROPS.get(material, {"E": 200e9, "alpha": 1.17e-5, "Sy": 250e6})
            E_pa = props["E"]
            alpha = props["alpha"]
            Sy = props["Sy"]
            
            st.markdown("---")
            
            # --- 1) 파열(Bursting) 내압 안전성 (Hoop Stress) ---
            st.markdown("### 💥 1. 배관 튜브 내압 파열(Bursting) 진단")
            
            # 관경과 두께 규명
            if d_input_method == "직접 입력":
                # 직접 입력 시 두께 추정 불가능 -> 보수적으로 내경의 10%를 방어벽(Thickness)으로 임시 가정
                t_m = D * 0.1
                od_m = D + 2*t_m
                st.warning("⚠️ **'직접 입력' 모드 동작 중:** 정확한 배관 두께를 알 수 없어 관 직경의 10%를 관벽 두께(t)로 임의 가정한 매우 보수적인 진단입니다. 실제 파열 한계를 명확하게 보려면 [사이드바]에서 **KS 규격 배관**을 선택하세요.")
            else:
                t_m = pipe_t / 1000.0
                od_m = pipe_od / 1000.0
                
            # 배관에 작용하는 최대 내부 압력 (기본 펌프 양정 압력 손실 + 수격 스파이크 반영)
            # (대기압 대비 상대압)
            max_op_pressure = dp_total * surge_multiplier
            
            if t_m > 0 and od_m > 0 and max_op_pressure > 0:
                # 얇은 관 실린더의 후프 응력(원주 방향 응력) 식: Barlow's formula -> Sigma_h = (P * OD) / (2 * t)
                hoop_stress = (max_op_pressure * od_m) / (2 * t_m)
                allowable_stress = Sy / safety_factor
                
                burst_sf = Sy / hoop_stress if hoop_stress > 0 else float('inf')
                
                st.write(f"▶ **산출된 배관 벽면 걸림 응력 (Hoop Stress):** **{hoop_stress/1e6:.2f} MPa**")
                st.write(f"▶ **재질의 허용 보호 응력 (Allowable Stress):** **{allowable_stress/1e6:.2f} MPa** (안전율 {safety_factor} 적용)")
                
                if hoop_stress < allowable_stress:
                    st.success(f"✅ **[안전] 파열 위험 없음:** 펌프의 최대 압송 부하와 밸브 충격(수격)이 발생하더라도 배관 관벽이 충분히 버틸 수 있습니다. (현재 극한 파괴 안전율: {burst_sf:.1f})")
                else:
                    st.error(f"🚨 **[위험] 내압 파열 한계 초과:** 시스템에 작용하는 압력이 너무 큽니다. 배관이 환경 노출이나 대기압 차이에 의해 터질(Burst) 위험이 있습니다! 더 두꺼운 SCH 규격을 사용하거나 펌프 용량을 분산시키세요.")
            elif dp_total <= 0:
                 st.info("배관 내 압력 변화가 거의 없거나 유속이 없어 파열을 진단할 내압이 존재하지 않습니다.")

            st.markdown("---")
            
            # --- 2) 열응력(Thermal Stress) 및 팽창 안전성 ---
            st.markdown("### 🌡️ 2. 한반도 4계절 온도 편차에 의한 열응력 및 열팽창 진단")
            st.write(f"배관 길이 **{L}m**, 배관 재질: **{material}**")
            st.caption(f"(재질 열팽창계수: {alpha:e} /°C, 탄성계수: {E_pa/1e9:.1f} GPa, 항복강도: {Sy/1e6:.1f} MPa)")
            
            # 유무체 열전도를 감안, 최악의 시나리오를 구성하기 위해 유체 온도와 외기 환경 온도 중 극한점을 도출
            T_fluid = temp_c if fluid_display != "직접 입력 (Manual)" and 'temp_c' in locals() else 20.0
            
            max_T = max(max_env_temp, T_fluid)
            min_T = min(min_env_temp, T_fluid)
            
            delta_T_exp = max_T - install_temp  # 팽창 온도폭
            delta_T_con = install_temp - min_T  # 수축 온도폭
            
            max_delta_T = max(abs(delta_T_exp), abs(delta_T_con))
            
            # 자유단 팽창량 (구속이 없을 때 파이프가 늘어나는 길이)
            delta_L = alpha * L * max_delta_T
            
            # 양단 완전 고정 시 열응력 (Thermal Stress = E * alpha * delta_T)
            thermal_stress = E_pa * alpha * max_delta_T
            
            col_th1, col_th2 = st.columns(2)
            with col_th1:
                st.markdown(f"**📏 최대 선팽창/수축 변형량**")
                st.markdown(f"<h3 style='color:#0F172A; margin:0;'>{delta_L * 1000:.1f} mm</h3>", unsafe_allow_html=True)
                st.caption(f"극한 온도 변화: {max_delta_T:.1f}°C")
                
                if delta_L > L * 0.001:
                    st.warning("⚠️ **[경고]** 파이프 길이가 1m당 1mm 이상 팽창/수축합니다. 일반적인 볼트 체결부가 파손될 수 있으므로 **신축 이음(Expansion Joint)이나 신축관 루프(U-Loop)** 설치 공간을 반드시 설계에 반영해야 합니다.")
                else:
                    st.success("✅ **[안전]** 팽창량이 미미하여 관의 자가 흡수 한도 내에서 관리가 가능한 수준입니다.")
                    
            with col_th2:
                st.markdown(f"**🧨 완전 고정 시 최대 잠재 열응력**")
                st.markdown(f"<h3 style='color:#0F172A; margin:0;'>{thermal_stress/1e6:.1f} MPa</h3>", unsafe_allow_html=True)
                
                # 열응력이 항복강도를 넘는가? (재질 파괴 마진율 1.5 적용)
                if thermal_stress > (Sy / 1.5):
                    st.error(f"🚨 **[위험] 응력 파괴 집중:** 지지대(Anchor)로 파이프 양단을 완전히 고정할 경우, 한여름 팽창이나 한겨울 수축으로 인한 응력({thermal_stress/1e6:.1f} MPa)이 재질 항복강도 안전선을 터무니없이 초과하여 배관이 휘거나 지지대가 뜯겨져 나갑니다! 이를 완화하는 플렉서블 튜브가 필수입니다.")
                else:
                    st.success("✅ **[안전]** 배관을 콘크리트에 완전히 매립·고정하더라도, 재질 자체가 여름/겨울철 열응력을 구조적으로 버틸 수 있어 파손되지 않습니다.")
                    
            st.markdown("</div>", unsafe_allow_html=True)
            
        with tab5:
            st.markdown("<div class='res-card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-header'>🔀 다중 관로(직렬/병렬) 네트워크 분석</div>", unsafe_allow_html=True)
            st.info("💡 배관이 도중에 넓어지거나 좁아지는 **직렬(Series)** 구간, 또는 두 갈래로 나뉘는 **병렬(Parallel)** 구간에 대한 유량 및 압력강하를 계산합니다.")
            
            network_type = st.radio("다중 관로 연결 방식", ["직렬 (Series) - 관경/저항이 직렬로 연결됨", "병렬 (Parallel) - 유량이 두 갈래로 분기됨"], horizontal=True)
            
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.markdown("### 🅰️ 배관 A 제원")
                L_A = st.number_input("배관 A 직관 길이 (m)", min_value=0.1, value=L, step=5.0)
                D_A_mm = st.number_input("배관 A 내경 (mm)", min_value=1.0, value=D*1000.0, step=5.0)
                K_A = st.number_input("배관 A 추가 저항계수 (K)", min_value=0.0, value=sum_K_fit + sum_K_valve, step=0.5)
                
            with col_p2:
                st.markdown("### 🅱️ 배관 B 제원")
                L_B = st.number_input("배관 B 직관 길이 (m)", min_value=0.1, value=L, step=5.0)
                D_B_mm = st.number_input("배관 B 내경 (mm)", min_value=1.0, value=D*1000.0*0.8, step=5.0, help="기본값으로 배관 A보다 20% 좁은 관경을 제시합니다.")
                K_B = st.number_input("배관 B 추가 저항계수 (K)", min_value=0.0, value=sum_K_fit + sum_K_valve, step=0.5)
                
            D_A = D_A_mm / 1000.0
            D_B = D_B_mm / 1000.0
            pipe_A = {"L": L_A, "D": D_A, "K": K_A}
            pipe_B = {"L": L_B, "D": D_B, "K": K_B}
            
            st.markdown("---")
            
            if "직렬" in network_type:
                # 직렬 배관 계산 로직
                v_A = calc_velocity(Q_m3s, D_A)
                Re_A = calc_reynolds(rho, v_A, D_A, mu)
                f_A, _ = calc_friction_factor(Re_A, D_A, epsilon)
                _, _, _, dp_A = calc_pressure_dp(f_A, L_A, D_A, rho, v_A, K_A, 0.0)
                
                v_B = calc_velocity(Q_m3s, D_B)
                Re_B = calc_reynolds(rho, v_B, D_B, mu)
                f_B, _ = calc_friction_factor(Re_B, D_B, epsilon)
                _, _, _, dp_B = calc_pressure_dp(f_B, L_B, D_B, rho, v_B, K_B, 0.0)
                
                dp_total_series = dp_A + dp_B
                
                st.markdown("<h3 style='color:#1E3A8A;'>🔗 직렬 (Series) 해석 결과</h3>", unsafe_allow_html=True)
                st.write(f"총 시스템 통과 유량: **{Q_m3s*60000:.1f} L/min** (배관 A, B 동일)")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("배관 A 압력강하", f"{dp_A/1e5:.4f} bar", f"유속 {v_A:.2f} m/s", delta_color="off")
                c2.metric("배관 B 압력강하", f"{dp_B/1e5:.4f} bar", f"유속 {v_B:.2f} m/s", delta_color="off")
                c3.metric("직렬 총 압력강하 (A+B)", f"{dp_total_series/1e5:.4f} bar", f"기본 조건 대비 {(dp_total_series - dp_total)/1e5:+.4f} bar", delta_color="inverse")
                
                st.progress(dp_A / dp_total_series if dp_total_series > 0 else 0, text="🟦 배관 A 저항 지분")
                st.progress(dp_B / dp_total_series if dp_total_series > 0 else 0, text="🟧 배관 B 저항 지분")
                
            else:
                # 병렬 배관 계산 로직
                Q_A, Q_B, dp_parallel = solve_parallel_2pipes(Q_m3s, pipe_A, pipe_B, rho, mu, epsilon)
                
                v_A = calc_velocity(Q_A, D_A)
                v_B = calc_velocity(Q_B, D_B)
                
                ratio_A = (Q_A / Q_m3s) * 100.0 if Q_m3s > 0 else 0
                ratio_B = (Q_B / Q_m3s) * 100.0 if Q_m3s > 0 else 0
                
                st.markdown("<h3 style='color:#1E3A8A;'>🔀 병렬 (Parallel) 분기 해석 결과</h3>", unsafe_allow_html=True)
                st.write(f"총 시스템 통과 유량: **{Q_m3s*60000:.1f} L/min**")
                
                c1, c2, c3 = st.columns(3)
                c1.metric("배관 A 통과 유량", f"{Q_A*60000:.1f} L/min", f"전체 유량의 {ratio_A:.1f}%", delta_color="off")
                c2.metric("배관 B 통과 유량", f"{Q_B*60000:.1f} L/min", f"전체 유량의 {ratio_B:.1f}%", delta_color="off")
                c3.metric("병렬 구간 압력강하", f"{dp_parallel/1e5:.4f} bar", f"기본 조건 대비 {(dp_parallel - dp_total)/1e5:+.4f} bar", delta_color="inverse")
                
                st.write("▪ 배관 A 유속: **{:.2f} m/s** | 배관 B 유속: **{:.2f} m/s**".format(v_A, v_B))
                
                if ratio_A > 90 or ratio_B > 90:
                    st.warning("⚠️ **편류 경고:** 어느 한쪽 배관으로 유량이 90% 이상 쏠려 있습니다. 유동 저항(길이, 관경, 밸브) 차이가 너무 크지 않은지 확인하세요.")
                else:
                    st.success("✅ 안정적인 유량 분배가 이루어지고 있습니다.")
                    
            st.markdown("</div>", unsafe_allow_html=True)
        
    else:
        # 계산 전 초기화면 안내
        st.markdown("""
        <div style='text-align:center; padding: 3rem 2rem; color:#5D6D7E; background:white; border-radius:12px; box-shadow:0 4px 6px rgba(0,0,0,0.05);'>
            <div style='font-size:4rem; margin-bottom:1rem;'>🛠️</div>
            <h3 style='color:#1E3A8A;'>좌측 사이드바에서 유체 및 시스템 조건을 입력하고<br>
            <b>[🚀 전체 시스템 시뮬레이션 결과 보기]</b> 버튼을 누르세요.</h3>
            <p style='margin-top:1rem; font-size:1rem; line-height: 1.6;'>
                본 프로그램은 단순한 직선 배관 마찰뿐만 아니라 <b>피팅 및 밸브에 의한 손실(Minor & Valve Losses)</b>을 명확하게 분리하여 분석하고,<br>
                유체 이송에 필요한 <b>펌프 요구 동력(Pump Power)</b>을 전력(kW) 단위로 정교하게 도출해 드립니다.
            </p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    build_ui()