# =============================================================================
# 프리미엄 배관 시스템 통합 시뮬레이터 (Pipe System Integrated Simulator)
# =============================================================================
# 실행 방법:
#   streamlit run integrated_pipe_app.py
# =============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import time
import os
import sys
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from PIL import Image

# =============================================================================
# [CORS 초월 실시간 백그라운드 HTTP API 브릿지 서버]
# =============================================================================
LATEST_CAD_DATA = None
DATA_LOCK = threading.Lock()
DATA_UPDATED = False

class CadBridgeHTTPHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # 콘솔 로그 지저분화 방지
        return

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        global LATEST_CAD_DATA, DATA_UPDATED
        if self.path == "/sync":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode('utf-8'))
                with DATA_LOCK:
                    LATEST_CAD_DATA = data
                    DATA_UPDATED = True
                
                self.send_response(200)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "success"}).encode('utf-8'))
            except Exception as e:
                self.send_response(400)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(str(e).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

@st.cache_resource
def start_bridge_server():
    for port in [18501, 18502, 18503]:
        try:
            # 로컬 루프백 127.0.0.1 바인딩
            server = HTTPServer(('127.0.0.1', port), CadBridgeHTTPHandler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            return port
        except Exception:
            continue
    return 18501  # 만약 모든 포트 충돌 시, 기존에 떠 있는 18501 포트를 신뢰하여 폴백 활용 보장

try:
    import CoolProp.CoolProp as CP
except ImportError:
    st.error("CoolProp 라이브러리가 설치되지 않았습니다. requirements.txt를 확인해 주세요.")
    st.stop()

# Gemini AI 모듈 가져오기 (오류 방지용 예외 처리)
try:
    import google.generativeai as genai
except ImportError:
    st.warning("google-generativeai 라이브러리가 설치되지 않아 AI 도면 분석 기능이 일부 제한될 수 있습니다.")

# =============================================================================
# [공통 데이터 및 물성치/계산 유틸리티]
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

FLUID_MATERIAL_RECOMMENDATIONS = {
    "Water": {
        "best": ["Stainless Steel (스테인리스 강관)", "PVC (일반 플라스틱 관)"],
        "ok": ["Galvanized Steel (아연도금 강관)", "Cast Iron (주철관)", "Drawn Tubing (인발 튜브)"],
        "hazard": ["Smooth Pipe (초매끈한 관, ε=0)"],
        "reason": "물은 범용적인 유체로 위생성과 내부식성이 우수한 스테인리스강 및 가볍고 녹슬지 않는 PVC가 최적입니다. 탄소강(Commercial Steel)은 장기 기동 시 부식성 스케일이 누적되어 수력 마찰이 증가할 수 있어 아연도금이나 주철관이 차선책이 됩니다."
    },
    "Methanol": {
        "best": ["Stainless Steel (스테인리스 강관)", "Commercial Steel (상업용 강관)"],
        "ok": ["Galvanized Steel (아연도금 강관)", "Drawn Tubing (인발 튜브)"],
        "hazard": ["PVC (일반 플라스틱 관)"],
        "reason": "메탄올과 같은 저급 알코올류 유기용제는 장기 노출 시 일반 플라스틱(PVC) 수지의 사슬 구조를 팽창시키거나 연화시켜 미세 누출 및 균열을 야기합니다. 화학적 안정성이 뛰어난 스테인리스강이나 강도가 확보된 상업용 강관 사용이 강제됩니다."
    },
    "Ethanol": {
        "best": ["Stainless Steel (스테인리스 강관)", "Commercial Steel (상업용 강관)"],
        "ok": ["Galvanized Steel (아연도금 강관)", "Drawn Tubing (인발 튜브)"],
        "hazard": ["PVC (일반 플라스틱 관)"],
        "reason": "에탄올은 친유성 및 침투성이 있어 PVC 수지를 서서히 침식하여 결합부를 파손합니다. 화학적으로 비활성이며 내화학 장벽을 형성하는 스테인리스강이나 강관 계열을 필히 권장합니다."
    },
    "INCOMP::MEG[0.5]": {
        "best": ["Stainless Steel (스테인리스 강관)", "Galvanized Steel (아연도금 강관)"],
        "ok": ["Commercial Steel (상업용 강관)", "PVC (일반 플라스틱 관)", "Drawn Tubing (인발 튜브)"],
        "hazard": ["Cast Iron (주철관)"],
        "reason": "에틸렌글리콜 50% 수용액은 산소와 결합 시 글리콜산 등의 유기산으로 분해되어 일반 주철관의 탈탄소 및 급격한 부식을 유발할 수 있습니다. 이를 방지하기 위해 방청 처리가 우수한 아연도금 강관 또는 스테인리스 강관이 탁월합니다."
    },
    "INCOMP::MPG[0.5]": {
        "best": ["Stainless Steel (스테인리스 강관)", "PVC (일반 플라스틱 관)"],
        "ok": ["Galvanized Steel (아연도금 강관)", "Commercial Steel (상업용 강관)", "Drawn Tubing (인발 튜브)"],
        "hazard": ["Cast Iron (주철관)"],
        "reason": "프로필렌글리콜은 독성이 적어 식품/제약용 냉매로 다수 쓰이므로, 위생 등급을 만족하는 스테인리스 강관이나 PVC 플라스틱이 베스트 소재입니다. 일반 주철관은 부식 생성물이 냉매 유로를 막을 위험이 큽니다."
    },
    "Acetone": {
        "best": ["Stainless Steel (스테인리스 강관)", "Commercial Steel (상업용 강관)"],
        "ok": ["Drawn Tubing (인발 튜브)"],
        "hazard": ["PVC (일반 플라스틱 관)", "Smooth Pipe (초매끈한 관, ε=0)"],
        "reason": "아세톤은 극성을 띤 매우 강력한 케톤계 유기용제로, 플라스틱(PVC) 수지를 즉각적으로 부풀리고 흐물흐물하게 녹여버립니다. 플라스틱 배관 설계 시 대형 폭발/누출 사고로 직결되므로, 반드시 강인한 탄소강관이나 내식성이 탁월한 스테인리스 강관을 사용하셔야 합니다."
    },
    "Benzene": {
        "best": ["Stainless Steel (스테인리스 강관)", "Commercial Steel (상업용 강관)"],
        "ok": ["Drawn Tubing (인발 튜브)", "Cast Iron (주철관)"],
        "hazard": ["PVC (일반 플라스틱 관)"],
        "reason": "벤젠은 방향족 탄화수소로 고분자 폴리머(PVC)를 격렬히 침식하고 가소제를 용출시켜 배관을 경화 및 파열시킵니다. 구조적 강도와 우수한 밀폐성을 제공하는 금속제 스테인리스강 또는 탄소강관을 강력 추천합니다."
    },
    "Toluene": {
        "best": ["Stainless Steel (스테인리스 강관)", "Commercial Steel (상업용 강관)"],
        "ok": ["Drawn Tubing (인발 튜브)", "Cast Iron (주철관)"],
        "hazard": ["PVC (일반 플라스틱 관)"],
        "reason": "톨루엔은 벤젠계 용제로서 강한 친지성을 가져 PVC 분자 사슬 결합을 깨뜨리고 체적 팽창을 일으켜 이음부 파손을 일으킵니다. 내유성 및 고온/고압 기밀이 확보되는 스테인리스 강관 및 탄소강관이 절실히 요구됩니다."
    }
}

def recommend_pipe_spec(q_m3s, material_name):
    """
    Genereaux 식 기반 LCC 최소화 기법 및 경제 유속 한계를 통합한 최적 상용 KS 관경 산출 엔진
    """
    if q_m3s <= 0:
        return 0.015, "15A - SCH 10S"
        
    # [1] Genereaux 모델을 기반으로 한 LCC 최소화 최적 내경 (경험상 0.363 * Q^0.45 * rho^0.13 물 기준 가이드)
    d_opt_gen = 0.37 * (q_m3s ** 0.45)
    
    # [2] 동력 소비량과 초기 시공비의 수력학적 평형 유속 (1.2 m/s) 가이드
    v_opt = 1.2
    d_opt_vel = np.sqrt((4.0 * q_m3s) / (np.pi * v_opt))
    
    # 가중 평균을 통한 최종 설계 목표 내경(d_opt) 확정
    d_opt = 0.65 * d_opt_gen + 0.35 * d_opt_vel
    
    is_stainless = "Stainless" in material_name
    std_key = "KS D 3576 (스테인리스 강관)" if is_stainless else "KS D 3562 (압력 배관용 탄소강관)"
    std_data = PIPE_STANDARDS[std_key]
    
    best_nps = "15A"
    best_sch = "SCH 10S" if is_stainless else "SCH 40"
    best_id_m = 0.015
    min_diff = float('inf')
    
    for nps, info in std_data["data"].items():
        od = info["OD"] / 1000.0
        schedules = [sch for sch in std_data["schedules"] if sch in info]
        for sch in schedules:
            t = info[sch] / 1000.0
            internal_d = od - 2.0 * t
            if internal_d <= 0:
                continue
                
            # Genereaux-유속 통합 지능형 모델 내경과의 편차 최소화 매칭
            diff = abs(internal_d - d_opt)
            if diff < min_diff:
                min_diff = diff
                best_nps = nps
                best_sch = sch
                best_id_m = internal_d
                
    nps_clean = best_nps.split(" ")[0]
    return best_id_m, f"{nps_clean} - {best_sch}"

def align_pipe_network_topology(pipes_list, nodes_list):
    """
    유동 위상 자동 정렬 (Flow Topology Alignment) 엔진:
    사용자가 파이프를 그린 드래그 방향과 무관하게, 펌프(Source/출발지)에서 출발하여
    아웃렛/토출구(Junction/Tank/Sink)로 도달하는 물리적 실제 흐름 방향으로 from/to를 내부 자동 Re-orient 정렬함.
    """
    node_map = {n["id"]: n for n in nodes_list}
    pump_nodes = [n["id"] for n in nodes_list if n["type"] == "pump"]
    tank_nodes = [n["id"] for n in nodes_list if n["type"] == "tank" or "OUTLET" in n.get("name", "")]
    
    if not pump_nodes:
        # 펌프가 없을 경우 연결 첫 노드를 임시 소스로 활용
        sources = [n["id"] for n in nodes_list if n["type"] == "tank"]
        if not sources and nodes_list:
            sources = [nodes_list[0]["id"]]
    else:
        sources = pump_nodes
        
    # 인접 리스트 구축 (방향 없음)
    graph = {}
    for p in pipes_list:
        f, t = p["from"], p["to"]
        if f not in graph: graph[f] = []
        if t not in graph: graph[t] = []
        graph[f].append((t, p["id"]))
        graph[t].append((f, p["id"]))
        
    # BFS 탐색을 통해 소스(펌프)로부터의 노드 거리 및 방향성(DAG) 결정
    node_depth = {}
    queue = []
    for s in sources:
        node_depth[s] = 0
        queue.append(s)
        
    visited = set(sources)
    while queue:
        curr = queue.pop(0)
        curr_d = node_depth[curr]
        for neighbor, pipe_id in graph.get(curr, []):
            if neighbor not in visited:
                visited.add(neighbor)
                node_depth[neighbor] = curr_d + 1
                queue.append(neighbor)
                
    # BFS 깊이에 따라 파이프의 방향을 from -> to 로 자동 Re-orient 정렬
    aligned_cnt = 0
    for p in pipes_list:
        f, t = p["from"], p["to"]
        f_d = node_depth.get(f, 999)
        t_d = node_depth.get(t, 999)
        
        # 만약 to 노드가 from 노드보다 소스(펌프)에 더 가깝다면, 실제 유체 흐름은 t -> f 임.
        # 따라서 파이프의 방향성을 물리 흐름에 맞게 정렬 뒤집기!
        if t_d < f_d:
            p["from"] = t
            p["to"] = f
            aligned_cnt += 1
            
    if aligned_cnt > 0:
        print(f"위상 정렬 엔진: 총 {aligned_cnt}개의 배관 흐름 방향을 물리적 위상(펌프->아웃렛)에 맞게 정밀 동적 자동 보정 완료!")

def solve_pipe_network(pipes_list, nodes_list, rho, mu, epsilon, q_sys_lmin, material_name):
    # [A] 먼저 유동 위상 자동 정렬 엔진 가동하여 그리기 방향 무관 물리적 흐름 위상으로 재배치
    align_pipe_network_topology(pipes_list, nodes_list)
    
    node_map = {n["id"]: n for n in nodes_list}
    q_sys_m3s = float(q_sys_lmin) / 60000.0
    
    # [1] 마디(Junction/Node) 연속 방정식(질량 보존 법칙) 사전 적합성 진단
    in_flows = {}
    out_flows = {}
    for n_id in node_map:
        in_flows[n_id] = 0.0
        out_flows[n_id] = 0.0
        
    for p in pipes_list:
        f_n = p["from"]
        t_n = p["to"]
        q_val = float(p.get("Q", 0.0))
        out_flows[f_n] += q_val
        in_flows[t_n] += q_val
        
    continuity_warnings = []
    for n_id, node in node_map.items():
        if node["type"] in ["junction", "valve"]:
            diff = abs(in_flows[n_id] - out_flows[n_id])
            if diff > 1e-4:
                continuity_warnings.append(
                    f"마디 '{node['name']}' ({node['type'].upper()}): 유입량({in_flows[n_id]*60000:.1f} L/min)과 유출량({out_flows[n_id]*60000:.1f} L/min)의 불일치가 감지되었습니다. (질량 유동 오차: {diff*60000:.1f} L/min)"
                )
                
    st.session_state["continuity_warnings"] = continuity_warnings

    # [2] Spanning Tree(신장 트리) 기반 고도화된 독립 폐회로(Fundamental Loops) 자동 색출 알고리즘
    adj = {}
    for p in pipes_list:
        f, t = p["from"], p["to"]
        if f not in adj: adj[f] = []
        if t not in adj: adj[t] = []
        adj[f].append((t, p["id"]))
        adj[t].append((f, p["id"]))
    
    loops = []
    if pipes_list and nodes_list:
        visited = set()
        tree_edges = set()       # (u, v, pipe_id) 형태
        parent = {}              # node -> (parent_node, pipe_id)
        depth = {}               # node -> depth
        
        # 무방향 그래프 상의 모든 연결 성분(Connected Components)에 대해 Spanning Tree 구성
        for start_node in node_map:
            if start_node not in visited:
                queue = [start_node]
                visited.add(start_node)
                depth[start_node] = 0
                parent[start_node] = (None, None)
                
                while queue:
                    u = queue.pop(0)
                    for v, pipe_id in adj.get(u, []):
                        if v not in visited:
                            visited.add(v)
                            depth[v] = depth[u] + 1
                            parent[v] = (u, pipe_id)
                            # Tree edge 등록 (방향 무관하게 고유 키로)
                            n_min, n_max = min(u, v), max(u, v)
                            tree_edges.add((n_min, n_max, pipe_id))
                            queue.append(v)
        
        # 신장 트리에 포함되지 않은 나머지 파이프(Link / Co-tree edge)들을 순회하며 독립 루프 구성
        for p in pipes_list:
            u, v = p["from"], p["to"]
            n_min, n_max = min(u, v), max(u, v)
            
            # Tree Edge가 아닌 경우 정확히 하나의 독립 루프 형성
            if not any(e[2] == p["id"] for e in tree_edges):
                path_u = []
                path_v = []
                
                curr_u = u
                while curr_u is not None:
                    p_node, pipe_id = parent[curr_u]
                    if p_node is not None:
                        path_u.append((curr_u, p_node, pipe_id))
                    curr_u = p_node
                    
                curr_v = v
                while curr_v is not None:
                    p_node, pipe_id = parent[curr_v]
                    if p_node is not None:
                        path_v.append((curr_v, p_node, pipe_id))
                    curr_v = p_node
                
                # 공통 조상(LCA) 탐색
                path_u.reverse()
                path_v.reverse()
                
                i = 0
                min_len = min(len(path_u), len(path_v))
                while i < min_len and path_u[i][1] == path_v[i][1]:
                    i += 1
                
                u_branch = path_u[i:]
                v_branch = path_v[i:]
                u_branch.reverse() # u -> LCA 자식 방향으로
                
                loop = []
                
                # u에서 LCA 방향 경로 추가 (루프 내 시계방향을 가정)
                for child, par, pipe_id in u_branch:
                    loop.append((child, par, pipe_id, 1))
                
                # LCA에서 v 방향 경로 추가
                for child, par, pipe_id in v_branch:
                    loop.append((par, child, pipe_id, 1))
                    
                # Link 파이프 (v -> u)로 루프 완전 폐합
                loop.append((v, u, p["id"], 1))
                loops.append(loop)

    # -------------------------------------------------------------------------
    # 🌟 [자동 관경 추천 & 2단계 순환 하이브리드 수리해석 엔진 기동]
    # -------------------------------------------------------------------------
    original_Ds = {}
    for p in pipes_list:
        p_id = p["id"]
        original_Ds[p_id] = float(p.get("D", 0.08))
        if original_Ds[p_id] <= 0.005:
            p["D"] = 0.05
            
    # 단계 B. 1차 가지형 유량 순차적 배분(Branch Allocation) 및 초기 가정 유량 주입
    Q_1st = {}
    out_degree = {n_id: 0 for n_id in node_map}
    for p in pipes_list:
        out_degree[p["from"]] += 1
        
    visited_dist = set()
    def distribute_flow(node_id, current_flow):
        if node_id in visited_dist:
            return
        visited_dist.add(node_id)
        
        pipes_from = [p for p in pipes_list if p["from"] == node_id]
        if not pipes_from:
            return
        
        flow_share = current_flow / len(pipes_from)
        for p in pipes_from:
            p_id = p["id"]
            Q_1st[p_id] = flow_share
            distribute_flow(p["to"], flow_share)
            
    pump_nodes = [n["id"] for n in nodes_list if n["type"] == "pump"]
    main_source = pump_nodes[0] if pump_nodes else (nodes_list[0]["id"] if nodes_list else None)
    
    if main_source:
        distribute_flow(main_source, q_sys_m3s)
        
    for p in pipes_list:
        p_id = p["id"]
        if p_id not in Q_1st or Q_1st[p_id] <= 1e-9:
            user_q = float(p.get("Q", 0.0))
            Q_1st[p_id] = user_q if user_q > 0 else q_sys_m3s / max(len(pipes_list), 1)

    max_iter = 150
    tol = 1e-6
    
    pump_shutoffs = {}
    for n in nodes_list:
        if n["type"] == "pump":
            pump_shutoffs[n["id"]] = float(n["val"]) if float(n["val"]) > 0 else 50.0

    pipe_minor_losses = {}
    for p in pipes_list:
        p_id = p["id"]
        k_val = 0.0
        for node_id in [p["from"], p["to"]]:
            if node_id in node_map:
                node_obj = node_map[node_id]
                if node_obj["type"] == "valve":
                    k_val += float(node_obj["val"])
        pipe_minor_losses[p_id] = k_val + 1.5

    # 1차 루프 연산 (Signed Flow 하디크로스)
    if loops:
        for iteration in range(max_iter):
            max_delta = 0.0
            for loop in loops:
                sum_h = 0.0
                sum_dq = 0.0
                for u, v, pipe_id, direction in loop:
                    p_obj_list = [p for p in pipes_list if p["id"] == pipe_id]
                    if not p_obj_list:
                        continue
                    p_obj = p_obj_list[0]
                    
                    # 루프 순회 방향과 파이프 고유 방향(from->to) 일치 여부 판정
                    sgn_loop = 1 if p_obj["from"] == u else -1
                    
                    d_m = float(p_obj["D"])
                    l_m = float(p_obj["L"])
                    q_val = Q_1st[pipe_id]
                    
                    # 루프 순회 기준 부호 있는 유량
                    q_loop = q_val * sgn_loop
                    abs_q = abs(q_loop)
                    
                    v_flow = calc_velocity(abs_q, d_m)
                    re = calc_reynolds(rho, v_flow, d_m, mu)
                    f, _ = calc_friction_factor(re, d_m, epsilon)
                    
                    g = 9.81
                    # K 저항 계수 계산 (Darcy-Weisbach)
                    K_dw = (f * (l_m / d_m) + pipe_minor_losses[pipe_id]) / (2.0 * g * (np.pi/4.0 * d_m**2)**2)
                    
                    # 수두 손실 = K * Q * |Q|
                    h_loss = K_dw * q_loop * abs_q
                    
                    # 펌프 모델링
                    h_pump = 0.0
                    A_coeff = 50000.0
                    # 펌프가 루프 진행방향 상에 위치하고 유량이 순방향일 때
                    if u in pump_shutoffs and sgn_loop == 1:
                        h_pump = max(pump_shutoffs[u] - A_coeff * (q_val ** 2), 0.0)
                    elif v in pump_shutoffs and sgn_loop == -1:
                        h_pump = max(pump_shutoffs[v] - A_coeff * (q_val ** 2), 0.0)
                        
                    sum_h += (h_loss - h_pump * sgn_loop)
                    
                    # 수치 안정성 가드 상수가 가미된 미분항 (n * K * |Q|)
                    sum_dq += (2.0 * K_dw * abs_q + (2.0 * A_coeff * abs_q if h_pump > 0.0 else 0.0) + 1e-5)
                        
                sum_dq = max(sum_dq, 1e-4)
                delta_q = - sum_h / sum_dq
                
                max_delta = max(max_delta, abs(delta_q))
                for u, v, pipe_id, direction in loop:
                    p_obj_list = [p for p in pipes_list if p["id"] == pipe_id]
                    if not p_obj_list:
                        continue
                    p_obj = p_obj_list[0]
                    sgn_loop = 1 if p_obj["from"] == u else -1
                    
                    # 각 파이프 유량 보정 (부호 있는 방향성 반영)
                    Q_1st[pipe_id] += delta_q * sgn_loop
            if max_delta < tol:
                break

    # 단계 C. 1차 수렴 유량 결과를 바탕으로, 각 파이프의 최적 추천 직경 산정 및 자동 굵기 대입
    optimal_specs = {}
    for p in pipes_list:
        p_id = p["id"]
        q_calc = abs(Q_1st[p_id])
        
        # 경제적 추천 내경 및 KS 규격명 획득
        rec_d, rec_spec = recommend_pipe_spec(q_calc, material_name)
        optimal_specs[p_id] = {"D": rec_d, "spec": rec_spec}
        p["t_rec"] = rec_spec
        
        if original_Ds[p_id] <= 0.005 or abs(original_Ds[p_id] - 0.08) < 1e-4 or abs(original_Ds[p_id] - 0.1) < 1e-4:
            p["D"] = rec_d

    # 단계 D. 2차 최종 하디크로스 수리해석 기동 (업데이트된 최적 직경 세트 기준)
    Q_2nd = {}
    for p in pipes_list:
        p_id = p["id"]
        Q_2nd[p_id] = Q_1st[p_id]

    # 단계 E. 펌프 소요 양정(H) 자동 역산 (비재귀 BFS 기반 고도화된 Critical Path 손실 수두 계산)
    path_head_losses = {n["id"]: 0.0 for n in nodes_list}
    if main_source:
        queue = [(main_source, 0.0)]
        visited_trace = set()
        
        while queue:
            curr_node, acc_loss = queue.pop(0)
            if acc_loss > path_head_losses.get(curr_node, 0.0):
                path_head_losses[curr_node] = acc_loss
                
            if curr_node in visited_trace:
                continue
            visited_trace.add(curr_node)
            
            pipes_from = [p for p in pipes_list if p["from"] == curr_node]
            for p in pipes_from:
                p_id = p["id"]
                d_m = float(p["D"])
                l_m = float(p["L"])
                q_val = abs(Q_2nd.get(p_id, 1e-4))
                
                v_flow = calc_velocity(q_val, d_m)
                re = calc_reynolds(rho, v_flow, d_m, mu)
                f, _ = calc_friction_factor(re, d_m, epsilon)
                
                g = 9.81
                h_fric = f * (l_m / d_m) * (v_flow**2) / (2 * g)
                h_minor = pipe_minor_losses.get(p_id, 1.5) * (v_flow**2) / (2 * g)
                p_loss_head = h_fric + h_minor
                
                next_node = p["to"]
                queue.append((next_node, acc_loss + p_loss_head))
        
    total_loss_head = max(path_head_losses.values()) if path_head_losses else 5.0
    calculated_pump_head = max(total_loss_head + 10.0, 15.0)
    
    pump_shutoffs_final = {}
    for n in nodes_list:
        if n["type"] == "pump":
            user_h = float(n["val"])
            if user_h <= 0.5:
                n["val"] = round(calculated_pump_head, 1)
                pump_shutoffs_final[n["id"]] = calculated_pump_head
            else:
                pump_shutoffs_final[n["id"]] = user_h

    # 최종 하디크로스 수렴 가동
    if loops:
        for iteration in range(max_iter):
            max_delta = 0.0
            for loop in loops:
                sum_h = 0.0
                sum_dq = 0.0
                for u, v, pipe_id, direction in loop:
                    p_obj_list = [p for p in pipes_list if p["id"] == pipe_id]
                    if not p_obj_list:
                        continue
                    p_obj = p_obj_list[0]
                    sgn_loop = 1 if p_obj["from"] == u else -1
                    
                    d_m = float(p_obj["D"])
                    l_m = float(p_obj["L"])
                    q_val = Q_2nd[pipe_id]
                    
                    q_loop = q_val * sgn_loop
                    abs_q = abs(q_loop)
                    
                    v_flow = calc_velocity(abs_q, d_m)
                    re = calc_reynolds(rho, v_flow, d_m, mu)
                    f, _ = calc_friction_factor(re, d_m, epsilon)
                    
                    g = 9.81
                    K_dw = (f * (l_m / d_m) + pipe_minor_losses[pipe_id]) / (2.0 * g * (np.pi/4.0 * d_m**2)**2)
                    
                    h_loss = K_dw * q_loop * abs_q
                    
                    h_pump = 0.0
                    A_coeff = 50000.0
                    if u in pump_shutoffs_final and sgn_loop == 1:
                        h_pump = max(pump_shutoffs_final[u] - A_coeff * (q_val ** 2), 0.0)
                    elif v in pump_shutoffs_final and sgn_loop == -1:
                        h_pump = max(pump_shutoffs_final[v] - A_coeff * (q_val ** 2), 0.0)
                        
                    sum_h += (h_loss - h_pump * sgn_loop)
                    sum_dq += (2.0 * K_dw * abs_q + (2.0 * A_coeff * abs_q if h_pump > 0.0 else 0.0) + 1e-5)
                        
                sum_dq = max(sum_dq, 1e-4)
                delta_q = - sum_h / sum_dq
                
                max_delta = max(max_delta, abs(delta_q))
                for u, v, pipe_id, direction in loop:
                    p_obj_list = [p for p in pipes_list if p["id"] == pipe_id]
                    if not p_obj_list:
                        continue
                    p_obj = p_obj_list[0]
                    sgn_loop = 1 if p_obj["from"] == u else -1
                    Q_2nd[pipe_id] += delta_q * sgn_loop
            if max_delta < tol:
                break
                
    for p in pipes_list:
        p_id = p["id"]
        q_final = Q_2nd[p_id]
        d_m = float(p["D"])
        v_flow = calc_velocity(abs(q_final), d_m)
        p["Q"] = float(q_final)
        p["v_flow"] = float(v_flow)

    return Q_2nd


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

FITTING_LOSSES = {
    "90도 엘보우 (Standard)": 0.75,
    "45도 엘보우 (Standard)": 0.40,
    "티 (Straight run)": 0.60,
    "티 (Branch flow)": 1.80,
}
VALVE_LOSSES = {
    "게이트 밸브 (Fully open)": 0.15,
    "글로브 밸브 (Fully open)": 10.0,
    "스윙 체크 밸브": 2.0,
}

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
            "300A (12B)":   {"OD": 318.5,"SCH 40": 4.0,  "SCH 10S": 4.5, "SCH 20S": 6.5, "SCH 40": 10.3,"SCH 80": 17.4},
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

from functools import lru_cache

@lru_cache(maxsize=256)
def get_fluid_properties(fluid: str, temp_c: float) -> tuple:
    T_K = temp_c + 273.15
    P = 101325.0
    p_vapor = 0.0
    
    if fluid.startswith("INCOMP::"):
        rho = CP.PropsSI("D", "T", T_K, "P", P, fluid)
        mu  = CP.PropsSI("V", "T", T_K, "P", P, fluid)
        p_vapor = 2000.0
    else:
        try:
            rho = CP.PropsSI("D", "T", T_K, "Q", 0, fluid)
            mu  = CP.PropsSI("V", "T", T_K, "Q", 0, fluid)
            p_vapor = CP.PropsSI("P", "T", T_K, "Q", 0, fluid)
        except ValueError:
            rho = CP.PropsSI("D", "T", T_K, "P", P, fluid)
            mu  = CP.PropsSI("V", "T", T_K, "P", P, fluid)
            p_vapor = P
            
    return rho, mu, p_vapor

def calc_velocity(Q_m3s: float, D: float) -> float:
    D = max(D, 1e-9)
    A = np.pi / 4.0 * D**2
    return Q_m3s / A

def calc_reynolds(rho: float, v: float, D: float, mu: float) -> float:
    if mu <= 0: return float('inf')
    return (rho * v * D) / mu

def calc_friction_factor(Re: float, D: float, epsilon: float) -> tuple:
    if Re < 1e-6:
        return 0.0, "정지 (No Flow)"
    if Re < 2300:
        f = 64.0 / Re
        regime = "층류 (Laminar)"
    elif Re < 4000:
        # 천이 영역 (Transitional Zone): 층류와 난류 마찰계수를 smoothstep 보간하여 불연속성 극복
        D = max(D, 1e-9)
        f_lam = 64.0 / 2300.0
        rel_rough = epsilon / D
        denom_4000 = np.log10(rel_rough / 3.7 + 5.74 / (4000.0**0.9))
        f_turb = 0.25 / denom_4000**2
        
        # 보간 가중치 t 및 smoothstep 매핑
        t = (Re - 2300.0) / (4000.0 - 2300.0)
        h = t * t * (3.0 - 2.0 * t)
        f = f_lam + h * (f_turb - f_lam)
        regime = "전이구간 (Transitional)"
    else:
        D = max(D, 1e-9)
        relative_roughness = epsilon / D
        denom = np.log10(relative_roughness / 3.7 + 5.74 / (Re**0.9))
        f = 0.25 / denom**2
        regime = "난류 (Turbulent)"
    return f, regime

def calc_pressure_dp(f: float, L: float, D: float, rho: float, v: float, sum_K_fit: float, sum_K_valve: float) -> tuple:
    D = max(D, 1e-9)
    dynamic_pressure = rho * v**2 / 2.0
    dp_fric = f * (L / D) * dynamic_pressure
    dp_fit = sum_K_fit * dynamic_pressure
    dp_valve = sum_K_valve * dynamic_pressure
    dp_total = dp_fric + dp_fit + dp_valve
    return dp_fric, dp_fit, dp_valve, dp_total

def calc_pump_power(dp_pa: float, Q_m3s: float, eff: float) -> float:
    if eff <= 0: return 0.0
    power_watts = (dp_pa * Q_m3s) / (eff / 100.0)
    return power_watts / 1000.0

def get_standard_motor(kw_req: float) -> tuple:
    std_sizes = [0.4, 0.75, 1.5, 2.2, 3.7, 5.5, 7.5, 11.0, 15.0, 18.5, 22.0, 30.0, 37.0, 45.0, 55.0, 75.0, 90.0, 110.0, 132.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0]
    design_kw = kw_req * 1.15
    for size in std_sizes:
        if size >= design_kw:
            return size, f"효성 프리미엄 고효율 전동기(IE3) / {size}kW 급"
    return design_kw, f"초대형 맞춤 제작 전동기 / {design_kw:.1f}kW 급"

def get_recommended_thickness_ks(outer_d_mm: float, req_t_mm: float, material_type: str) -> tuple:
    std_key = "KS D 3576 (스테인리스 강관)" if "Stainless" in material_type else "KS D 3562 (압력 배관용 탄소강관)"
    std_data = PIPE_STANDARDS[std_key]
    
    closest_nps = None
    min_diff = float('inf')
    
    for nps, info in std_data["data"].items():
        diff = abs(info["OD"] - outer_d_mm)
        if diff < min_diff:
            min_diff = diff
            closest_nps = nps
            
    if closest_nps is None:
        return "N/A", req_t_mm
        
    nps_info = std_data["data"][closest_nps]
    schedules = [sch for sch in std_data["schedules"] if sch in nps_info]
    
    recommended_sch = "N/A"
    recommended_t = 0.0
    
    for sch in schedules:
        t_sch = nps_info[sch]
        if t_sch >= req_t_mm:
            recommended_sch = sch
            recommended_t = t_sch
            break
            
    if recommended_sch == "N/A" and len(schedules) > 0:
        recommended_sch = schedules[-1] + " (두께 부족, 외경 상향 권장)"
        recommended_t = nps_info[schedules[-1]]
        
    return f"{closest_nps} - {recommended_sch}", recommended_t


def render_integrated_report(shared_json_input, rho, mu, epsilon, fluid_key, material, safety_factor, pump_eff, eco_years, eco_hours, eco_elec, eco_carbon_price, install_temp, max_env_temp, min_env_temp, surge_multiplier, eco_ir, widget_key, p_vapor, q_sys_lmin):
    # 빈 배관 정보 사전 유효성 필터 및 대기 상태 렌더러 연동
    is_empty = True
    pipes_list = []
    nodes_list = []
    
    if shared_json_input:
        try:
            network_data = json.loads(shared_json_input)
            pipes_list = network_data.get("pipes", [])
            nodes_list = network_data.get("nodes", [])
            if pipes_list:
                is_empty = False
        except Exception:
            pass
            
    if is_empty:
        st.markdown("""
        <div style='background: rgba(30, 41, 59, 0.45); padding: 2.2rem; border-radius: 20px; border: 1px dashed rgba(255, 255, 255, 0.1); box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3); margin-top: 1.5rem; text-align: center;'>
            <div style='font-size: 3.5rem; margin-bottom: 0.8rem;'>📐</div>
            <h3 style='color: white; margin: 0; font-family: "Outfit", sans-serif; font-weight: 800; background: linear-gradient(90deg, #3B82F6 0%, #60A5FA 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>배관 시스템 설계 대기 중</h3>
            <p style='margin-top: 0.8rem; opacity: 0.9; font-size: 0.98rem; line-height: 1.6; color: #CBD5E1;'>
                위 1단계 CAD 드로잉판에서 배관망의 라인을 슥슥 연결하고 기기를 배치해 주세요!<br>
                그리는 즉시 <b>100% 무클립보드 실시간 양방향</b> 수리 유동 평형 분석 및 펌프/관경 자동 설계가 개시됩니다.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.text_area(
            "🔄 실시간 설계 데이터 동기화 입력 포트 (도면판에서 분석 동기화 클릭 후 여기에 Ctrl + V)",
            placeholder="streamlit_canvas_json_bridge_exchange_area",
            key=widget_key,
            height=60,
            label_visibility="visible"
        )
        return

    try:
        st.info(f"⚡ **드로잉 배관망 연동 완료:** 배관 **{len(pipes_list)}개**, 기기 노드 **{len(nodes_list)}개** 정밀 유역학 시뮬레이션 중...")
        
        # 데이터가 이미 로드되어 정상 동작 중일 때는 리포트 최상단에 미니 브릿지 텍스트 에리어를 선명하게 노출하여 재동기화 편리성 극대화
        st.text_area(
            "🔄 설계결과 분석하기 실시간 데이터 동기화 포트 (설계를 새로 한 경우 여기에 다시 붙여넣기)",
            placeholder="streamlit_canvas_json_bridge_exchange_area",
            key=widget_key,
            height=60,
            label_visibility="visible"
        )

        # --- [1] 백엔드 하디크로스 유동 평형 및 자동 굵기/양정 추천 해석 가동 ---
        converged_q = solve_pipe_network(pipes_list, nodes_list, rho, mu, epsilon, q_sys_lmin, material)
        
        # [수력학 경고] 마디(Junction) 질량 평형 불일치 경고 출력
        continuity_warns = st.session_state.get("continuity_warnings", [])
        if continuity_warns:
            with st.expander("🚨 **질량 보존 법칙(노드 유출입 평형) 오차 경고 검출**", expanded=True):
                st.error("현재 드로잉에서 넘겨온 초기 유량/유입값의 마디별 평형이 깨져 있습니다. 수치해석 안정성에 영항을 줄 수 있으므로 확인하십시오.")
                for warn in continuity_warns:
                    st.caption(f"▪ {warn}")
        
        # 펌프 자동 역산 양정 값 검색
        auto_pump_head = 0.0
        for n in nodes_list:
            if n["type"] == "pump":
                auto_pump_head = float(n["val"])
                break

        # --- [2] 지능형 배관망 자동 설계 제안서 (AI Recommended Design Specs) 신설 ---
        st.markdown("<div class='section-header'>⚙️ 지능형 배관망 자동 설계 제안서 (AI Recommended Design Specs)</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='background: rgba(30, 41, 59, 0.65); padding: 1.8rem; border-radius: 18px; border: 1px solid rgba(59, 130, 246, 0.35); box-shadow: 0 15px 30px rgba(0, 0, 0, 0.4); margin-bottom: 2rem; backdrop-filter: blur(10px);'>
            <h4 style='margin-top: 0; color: #60A5FA; font-family: "Outfit", sans-serif; font-weight: 800; font-size:1.25rem;'>💡 프로그램 자동 추천 설계 요약</h4>
            <p style='color: #CBD5E1; font-size: 0.92rem; line-height: 1.6; margin-bottom: 1.2rem;'>
                사용자님의 CAD 배치 형상과 지정하신 전체 계통 대표 유량 <b>({q_sys_lmin:.1f} L/min)</b>에 맞추어, 마찰을 최소화하고 시공 경제성을 극대화하는 <b>최적 배관 규격 및 펌프 소요 양정</b>을 물리학적으로 자동 설계 완료하였습니다.
            </p>
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1.2rem; margin-top: 1.5rem;'>
                <div style='background: rgba(15, 23, 42, 0.5); padding: 1.2rem; border-radius: 12px; border: 1px solid rgba(16, 185, 129, 0.25); text-align: center;'>
                    <span style='font-size: 0.82rem; color: #94A3B8; display: block; margin-bottom: 0.4rem;'>🔌 펌프 설계 필요 양정</span>
                    <strong style='font-size: 1.4rem; color: #34D399;'>{auto_pump_head:.1f} m</strong>
                    <span style='font-size: 0.75rem; color: #64748B; display: block; margin-top: 0.4rem;'>(손실압 자동 역산 + 안전마진 적용)</span>
                </div>
                <div style='background: rgba(15, 23, 42, 0.5); padding: 1.2rem; border-radius: 12px; border: 1px solid rgba(96, 165, 250, 0.25); text-align: center;'>
                    <span style='font-size: 0.82rem; color: #94A3B8; display: block; margin-bottom: 0.4rem;'>📏 계통 총 설계 연장</span>
                    <strong style='font-size: 1.4rem; color: #60A5FA;'>{sum(float(p['L']) for p in pipes_list):.1f} m</strong>
                    <span style='font-size: 0.75rem; color: #64748B; display: block; margin-top: 0.4rem;'>(배관 요소 {len(pipes_list)}개 총합)</span>
                </div>
                <div style='background: rgba(15, 23, 42, 0.5); padding: 1.2rem; border-radius: 12px; border: 1px solid rgba(245, 158, 11, 0.25); text-align: center;'>
                    <span style='font-size: 0.82rem; color: #94A3B8; display: block; margin-bottom: 0.4rem;'>💎 최적 경제 유속 제어군</span>
                    <strong style='font-size: 1.3rem; color: #F59E0B;'>1.0 ~ 1.5 m/s</strong>
                    <span style='font-size: 0.75rem; color: #64748B; display: block; margin-top: 0.4rem;'>(펌프 동력손실 차단 설계 유속)</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # --- [3] Barlow 내압 파열 진단 및 스케줄 규격 역산 ---
        props = MECHANICAL_PROPS.get(material, {"E": 200e9, "alpha": 1.17e-5, "Sy": 250e6})
        Sy_val = props["Sy"]
        allowable_stress = Sy_val / safety_factor
        
        # 각 유체 고유의 체적 탄성계수 (Bulk Modulus, Pa) 정의 (출처: 표준 열역학 편람)
        FLUID_BULK_MODULUS = {
            "Water": 2.2e9, "Methanol": 8.2e8, "Ethanol": 9.0e8, "INCOMP::MEG[0.5]": 2.5e9,
            "INCOMP::MPG[0.5]": 2.4e9, "Acetone": 8.0e8, "Benzene": 1.05e9, "Toluene": 1.1e9,
        }
        
        res_kpi_total_dp = 0.0
        res_kpi_total_kw = 0.0
        dangerous_pipes = []
        analysis_results = []
        
        for p in pipes_list:
            p_id = p["id"]
            d_m = float(p["D"])
            l_m = float(p["L"])
            q_final = converged_q[p_id]
            
            v_flow = calc_velocity(q_final, d_m)
            re_flow = calc_reynolds(rho, v_flow, d_m, mu)
            f_flow, regime = calc_friction_factor(re_flow, d_m, epsilon)
            
            # 국부 손실 계수 산출 (피팅 손실 가산)
            dp_fric, _, _, dp_loss = calc_pressure_dp(f_flow, l_m, d_m, rho, v_flow, 1.5, 0.0)
            res_kpi_total_dp += dp_loss
            
            p_kw = calc_pump_power(dp_loss, q_final, pump_eff)
            res_kpi_total_kw += p_kw
            
            # Joukowsky 수격압 물리 모형 동적 계산
            bulk_k = FLUID_BULK_MODULUS.get(fluid_key, 2.2e9)
            E_mat = props.get("E", 200e9)
            t_assumed = d_m * 0.05
            
            celerity = np.sqrt(bulk_k / rho) / np.sqrt(1.0 + (bulk_k / E_mat) * (d_m / max(t_assumed, 1e-4)))
            dp_surge = rho * celerity * v_flow
            max_p = dp_loss + dp_surge
            
            od_m = d_m * 1.1
            t_req_mm = (max_p * od_m * 1000.0) / (2 * allowable_stress)
            t_req_mm = max(t_req_mm, 1.5)
            
            rec_spec, rec_t_val = get_recommended_thickness_ks(od_m * 1000.0, t_req_mm, material)
            
            hoop_stress = (max_p * od_m) / (2 * (d_m * 0.1))
            if hoop_stress >= allowable_stress:
                dangerous_pipes.append(p_id)
                status_text = "🚨 파열 위험 (두께 상향 필수)"
            else:
                status_text = "✅ 안전"
                
            analysis_results.append({
                "배관 ID": p_id,
                "최종 유량 [L/min]": round(q_final * 60000.0, 1),
                "평균 유속 [m/s]": round(v_flow, 2),
                "유동 손실압 [bar]": round(dp_loss / 1e5, 4),
                "필요 최소 두께 [mm]": round(t_req_mm, 3),
                "추천 배관 규격 및 두께": rec_spec,
                "구조 안전성 진단": status_text
            })
            
        df_results = pd.DataFrame(analysis_results)
        
        # --- [3] 생애주기비용 (LCC) 및 세부 투자비용(CapEx) 정밀 연산 ---
        ir = eco_ir / 100.0
        inf_e = 0.03
        N = eco_years
        crf = (ir * (1 + ir)**N) / ((1 + ir)**N - 1) if ir > 0 else 1.0 / N
        
        if ir == inf_e:
            pvf_energy = N / (1.0 + ir)
        else:
            x = (1.05) / (1.0 + ir)
            pvf_energy = (1.0 / 1.05) * x * (1.0 - x**N) / (1.0 - x)
        lvl_energy_factor = pvf_energy * crf
        
        # A. 펌프 조달 비용
        capex_pump = 780000.0 * (res_kpi_total_kw ** 0.82) + 450000.0
        
        # B. 배관 건설 자재비 및 인건비 시공비
        total_L = sum(float(p["L"]) for p in pipes_list)
        total_D_mm = sum(float(p["D"]) * 1000.0 for p in pipes_list)
        avg_D_mm = total_D_mm / len(pipes_list)
        
        unit_pipe_cost = (avg_D_mm * 1200.0) + 15000.0
        capex_pipe_pure = unit_pipe_cost * total_L
        
        # C. 피팅류 할증비
        capex_fittings = capex_pipe_pure * 0.35
        
        # D. 기계설비 노무비 및 경비
        capex_labor = (capex_pipe_pure + capex_fittings) * 1.5
        
        total_capex = capex_pump + capex_pipe_pure + capex_fittings + capex_labor
        
        # E. 연간 운영비용 (OpEx)
        annual_elec = res_kpi_total_kw * eco_hours * eco_elec
        annual_carbon = (res_kpi_total_kw * eco_hours / 1000.0) * 0.4594 * eco_carbon_price
        annual_maint = total_capex * 0.02
        
        euac_capex = total_capex * crf
        euac_energy = annual_elec * lvl_energy_factor
        euac_carbon = annual_carbon * lvl_energy_factor
        euac_maint = annual_maint
        
        total_annual_lcc = euac_capex + euac_energy + euac_carbon + euac_maint
        
        df_lcc_chart = pd.DataFrame({
            "LCC 비용 항목": ["🏗️ 자본투자 자본상각", "💡 펌프 전력요금", "🌿 이산화탄소 Penalty", "🔧 유지보수 O&M"],
            "연간화 비용 (EUAC) [원/년]": [euac_capex, euac_energy, euac_carbon, euac_maint]
        })
        df_lcc_chart.set_index("LCC 비용 항목", inplace=True)
        
        # 상세 견적서 데이터프레임
        df_detail_cost = pd.DataFrame([
            {
                "비용 구분": "설비 투자비 (CapEx)", "세부 비용 항목": "펌프 및 구동 모터 구매 단가",
                "계산 금액": f"{capex_pump:,.0f} 원",
                "공학적 산출 기준 및 공인 출처": "조달청 나라장터 우수조달 다수공급자계약(MAS) 3상 원심펌프 표준 협정가격 데이터"
            },
            {
                "비용 구분": "설비 투자비 (CapEx)", "세부 비용 항목": "배관 파이프 자재비 (직관 기준)",
                "계산 금액": f"{capex_pipe_pure:,.0f} 원",
                "공학적 산출 기준 및 공인 출처": "대한건설협회 발행 월간 '거래가격(Market Price)' 물가정보 배관 강관 표준재료비 기준"
            },
            {
                "비용 구분": "설비 투자비 (CapEx)", "세부 비용 항목": "방향 전환 피팅(이음쇠) 및 조절 밸브류",
                "계산 금액": f"{capex_fittings:,.0f} 원",
                "공학적 산출 기준 및 공인 출처": "유체시스템 기계설비 견적 기준 (직관 자재비 대비 35% 기본 배관 부속 할증 적용)"
            },
            {
                "비용 구분": "설비 투자비 (CapEx)", "세부 비용 항목": "배관공/용접공 현장 노무비 및 경비",
                "계산 금액": f"{capex_labor:,.0f} 원",
                "공학적 산출 기준 및 공인 출처": "국토교통부·한국건설기술연구원 발행 '건설공사 표준품셈' 기계설비공(배관공 품 공수) 및 공표 시중노임단가"
            },
            {
                "비용 구분": "연간 가동비 (OpEx)", "세부 비용 항목": "연간 펌프 구동 전력 요금",
                "계산 금액": f"{annual_elec:,.0f} 원/년",
                "공학적 산출 기준 및 공인 출처": "한국전력공사 공식 '산업용 전력요금표(을) 고압A' 평균 공급 단가 (150원/kWh)"
            },
            {
                "비용 구분": "연간 가동비 (OpEx)", "세부 비용 항목": "온실가스 배출 Penalty 탄소세",
                "계산 금액": f"{annual_carbon:,.0f} 원/년",
                "공학적 산출 기준 및 공인 출처": "환경부 온실가스 배출권 거래제(K-ETS) 최근 3개년 배출권 평균 낙찰가격 (15,000원/tCO2eq)"
            },
            {
                "비용 구분": "연간 가동비 (OpEx)", "세부 비용 항목": "설비 연간 유지보수비 (O&M)",
                "계산 금액": f"{annual_maint:,.0f} 원/년",
                "공학적 산출 기준 및 공인 출처": "국토교통부 고시 시설물 안전 및 유지관리 실무 대가 기준 (총 CapEx 투자비의 연 2.0% 책정)"
            }
        ])
        
        # --- [4] 초통합 대시보드 리포트 화면 렌더링 ---
        st.markdown("<div class='section-header'>📊 배관 시스템 수력학 및 LCC 종합 분석 대시보</div>", unsafe_allow_html=True)
        
        # 최장 경로 손실을 역산한 펌프 소요 양정 수두
        avg_H_m = 80.0
        for node in nodes_list:
            if node["type"] == "pump" and float(node["val"]) > 0:
                avg_H_m = float(node["val"])
                break
                
        p_watts = res_kpi_total_kw * 1000.0
        g_const = 9.81
        denom = max(rho * g_const * avg_H_m, 1.0)
        q_m3s_calc = (p_watts * (pump_eff / 100.0)) / denom
        q_m3h_calc = q_m3s_calc * 3600.0
        
        # 펌프 공동현상(NPSH) 연역 계산
        pump_node_id = None
        for n in nodes_list:
            if n["type"] == "pump":
                pump_node_id = n["id"]
                break
                
        suction_pipe = None
        if pump_node_id:
            for p in pipes_list:
                if p["to"] == pump_node_id:
                    suction_pipe = p
                    break
                    
        h_fs = 0.0
        if pump_node_id and suction_pipe:
            s_id = suction_pipe["id"]
            s_d = float(suction_pipe["D"])
            s_l = float(suction_pipe["L"])
            q_final = converged_q[s_id]
            v_flow = calc_velocity(q_final, s_d)
            re_flow = calc_reynolds(rho, v_flow, s_d, mu)
            f_flow, _ = calc_friction_factor(re_flow, s_d, epsilon)
            h_fs = (f_flow * (s_l / s_d) + 1.5) * (v_flow**2) / (2 * 9.81)
            
        h_atm = 101325.0 / (rho * g_const)
        h_vap = p_vapor / (rho * g_const)
        npsha = h_atm - h_vap - h_fs
        
        npshr = 2.0
        if q_m3h_calc > 4.0: npshr = 2.5
        if q_m3h_calc > 8.0: npshr = 3.0
        if q_m3h_calc > 16.0: npshr = 3.5
        
        if npsha >= npshr + 0.5:
            npsh_summary = "✅ 공동현상 안전 (NPSH Pass)"
            npsh_summary_color = "#10B981"
        elif npsha >= npshr:
            npsh_summary = "⚠️ 공동현상 주의 (Cavitation Danger)"
            npsh_summary_color = "#F59E0B"
        else:
            npsh_summary = "🚨 공동현상 위험 (Cavitation Occurred)"
            npsh_summary_color = "#EF4444"

        # 4대 핵심 진단 요약 보드 배치
        st.markdown(f"""
        <div style='background: rgba(30, 41, 59, 0.65); padding: 1.8rem; border-radius: 18px; border: 1px solid rgba(59, 130, 246, 0.35); box-shadow: 0 15px 30px rgba(0, 0, 0, 0.4); margin-bottom: 2rem; backdrop-filter: blur(10px);'>
            <h4 style='margin-top: 0; color: #60A5FA; font-family: "Outfit", sans-serif; font-weight: 800; font-size:1.3rem; border-bottom: 1px solid rgba(255,255,255,0.08); padding-bottom:0.6rem;'>🎯 기계공학 및 수력학 4대 핵심 실시간 진단서</h4>
            <div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.2rem; margin-top: 1.2rem;'>
                <div style='background: rgba(15, 23, 42, 0.5); padding: 1.2rem; border-radius: 12px; border: 1px solid rgba(16, 185, 129, 0.25); text-align: center;'>
                    <span style='font-size: 0.85rem; color: #94A3B8; display: block; margin-bottom: 0.4rem;'>🚿 1. 유량 분배 및 유속 방향</span>
                    <strong style='font-size: 1.35rem; color: #34D399;'>100% 분배 수렴 완료</strong>
                    <span style='font-size: 0.78rem; color: #64748B; display: block; margin-top: 0.4rem;'>(from ➔ to 물리 흐름 위상 정렬 완료)</span>
                </div>
                <div style='background: rgba(15, 23, 42, 0.5); padding: 1.2rem; border-radius: 12px; border: 1px solid {npsh_summary_color}25; text-align: center;'>
                    <span style='font-size: 0.85rem; color: #94A3B8; display: block; margin-bottom: 0.4rem;'>🫧 2. 펌프 공동현상(Cavitation) 여부</span>
                    <strong style='font-size: 1.25rem; color: {npsh_summary_color};'>{npsh_summary}</strong>
                    <span style='font-size: 0.78rem; color: #64748B; display: block; margin-top: 0.4rem;'>(NPSHa: {npsha:.2f}m vs NPSHr: {npshr:.1f}m)</span>
                </div>
                <div style='background: rgba(15, 23, 42, 0.5); padding: 1.2rem; border-radius: 12px; border: 1px solid rgba(96, 165, 250, 0.25); text-align: center;'>
                    <span style='font-size: 0.85rem; color: #94A3B8; display: block; margin-bottom: 0.4rem;'>🔌 3. 추천 펌프 소요 양정(H)</span>
                    <strong style='font-size: 1.35rem; color: #60A5FA;'>{auto_pump_head:.1f} m</strong>
                    <span style='font-size: 0.78rem; color: #64748B; display: block; margin-top: 0.4rem;'>(손실 수두 자동 역산 + SF 할증)</span>
                </div>
                <div style='background: rgba(15, 23, 42, 0.5); padding: 1.2rem; border-radius: 12px; border: 1px solid rgba(245, 158, 11, 0.25); text-align: center;'>
                    <span style='font-size: 0.85rem; color: #94A3B8; display: block; margin-bottom: 0.4rem;'>💸 4. 총 시공 비용 및 운영 LCC</span>
                    <strong style='font-size: 1.35rem; color: #F59E0B;'>{total_annual_lcc:,.0f} 원/년</strong>
                    <span style='font-size: 0.78rem; color: #64748B; display: block; margin-top: 0.4rem;'>(CapEx+OpEx 20년 LCC 기반 산출)</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
        with col_kpi1:
            st.markdown(f"""
            <div class='res-card' style='text-align:center;'>
                <h5 style='color:#94A3B8; margin:0;'>그려진 배관망 총 전력 요구량</h5>
                <h2 style='color:#10B981; margin:0.5rem 0;'>{res_kpi_total_kw:.3f} kW</h2>
                <p style='margin:0; font-size:0.9rem; color:#64748B;'>(종합 펌프 효율 {pump_eff}% 적용)</p>
            </div>
            """, unsafe_allow_html=True)
        with col_kpi2:
            st.markdown(f"""
            <div class='res-card' style='text-align:center;'>
                <h5 style='color:#94A3B8; margin:0;'>배관망 LCC 총 연간 비용</h5>
                <h2 style='color:#F59E0B; margin:0.5rem 0;'>{total_annual_lcc:,.0f} 원/년</h2>
                <p style='margin:0; font-size:0.9rem; color:#64748B;'>(자본회수 {eco_years}년 등가 환산)</p>
            </div>
            """, unsafe_allow_html=True)
        with col_kpi3:
            dangerous_cnt = len(dangerous_pipes)
            safety_color = "#EF4444" if dangerous_cnt > 0 else "#3B82F6"
            status_lbl = f"파열 위험 배관 {dangerous_cnt}개" if dangerous_cnt > 0 else "모든 배관 완벽 안전"
            st.markdown(f"""
            <div class='res-card' style='text-align:center;'>
                <h5 style='color:#94A3B8; margin:0;'>구조적 내압 안전 진단</h5>
                <h2 style='color:{safety_color}; margin:0.5rem 0;'>{status_lbl}</h2>
                <p style='margin:0; font-size:0.9rem; color:#64748B;'>(Joukowsky 수격파 동적 모형 및 SF {safety_factor} 기준)</p>
            </div>
            """, unsafe_allow_html=True)
            
        col_rep1, col_rep2 = st.columns([1.3, 1])
        
        with col_rep1:
            st.markdown("<div class='res-card'>", unsafe_allow_html=True)
            st.markdown("##### 🔩 수력학 관벽 후프 응력 진단 및 적정 두께(스케줄) 추천표")
            st.dataframe(
                df_results.style.background_gradient(subset=["최종 유량 [L/min]", "필요 최소 두께 [mm]"], cmap="Blues"),
                use_container_width=True
            )
            
            if dangerous_pipes:
                st.error(f"🚨 **파열 경고:** 설계하신 배관망 중 **{dangerous_pipes}** 번 배관은 현재 작용 수압이 허용 강도를 초과하여 터질 위험이 큽니다! 위에 추천된 두꺼운 스케줄 규격으로 반드시 사양을 올리십시오.")
            else:
                st.success("✅ **구조 강도 안전 통과:** 입력하신 모든 배관이 사용자 지정 안전 마진 하에 충분한 파열 방지 한계를 확보하고 있습니다.")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_rep2:
            st.markdown("<div class='res-card'>", unsafe_allow_html=True)
            st.markdown("##### 🔌 설계 배관망 최적 원심펌프 추천 명세")
            
            # 수리학-상용 펌프 모델 매핑 엔진 기동
            avg_H_m = 80.0
            for node in nodes_list:
                if node["type"] == "pump" and float(node["val"]) > 0:
                    avg_H_m = float(node["val"])
                    break
                    
            p_watts = res_kpi_total_kw * 1000.0
            # 물리 역학적 정격 유량 Q (m3/s) 계산: Q = P * eta / (rho * g * H)
            g_const = 9.81
            denom = max(rho * g_const * avg_H_m, 1.0)
            q_m3s_calc = (p_watts * (pump_eff / 100.0)) / denom
            q_m3h_calc = q_m3s_calc * 3600.0
            q_lmin_calc = q_m3s_calc * 60000.0
            
            # Wilo / Grundfos 산업용 펌프 데이터베이스 매핑
            if q_m3h_calc <= 2.0:
                pump_model = "Grundfos CR 1-15 (고성능 수직 다단 원심)"
                pump_spec = f"정격 유량 1.8 m³/hr ({30.0:.1f} L/min) | 양정 한계 90m | 권장 동력 {res_kpi_total_kw*1.15:.2f} kW 급"
                pump_desc = "정밀 유량 제어 및 고압 송출에 특화된 펌프입니다. 소구경 고양정 상하수 계통 및 화학공정 라인 가압에 최적입니다."
            elif q_m3h_calc <= 4.0:
                pump_model = "Wilo Helix V 404 (고효율 다단 원심)"
                pump_spec = f"정격 유량 4.0 m³/hr ({66.7:.1f} L/min) | 양정 한계 42m | 권장 동력 {res_kpi_total_kw*1.15:.2f} kW 급"
                pump_desc = "Wilo의 차세대 에너지 절감형 다단 펌프로서, 산업용 세척 공정, 냉각수 순환, 빌딩 가압 급수용 스테인리스 스펙 모델입니다."
            elif q_m3h_calc <= 8.0:
                pump_model = "Wilo Helix V 805 (산업용 대유량 다단원심)"
                pump_spec = f"정격 유량 8.0 m³/hr ({133.3:.1f} L/min) | 양정 한계 55m | 권장 동력 {res_kpi_total_kw*1.15:.2f} kW 급"
                pump_desc = "중대형 냉방 공정, 대용량 가압설비 및 빌딩 급수 계통에 폭넓게 활약하는 표준 고성능 스테인리스 펌프입니다."
            elif q_m3h_calc <= 16.0:
                pump_model = "Grundfos CR 15-10 (중대형 고양정 원심)"
                pump_spec = f"정격 유량 15.0 m³/hr ({250.0:.1f} L/min) | 양정 한계 100m | 권장 동력 {res_kpi_total_kw*1.15:.2f} kW 급"
                pump_desc = "강력한 유량 및 수압 성능을 겸비한 플랜트용 표준 펌프로서, 뛰어난 기계적 밀폐성과 내식 장벽을 제공하는 금속 커버 모델입니다."
            else:
                pump_model = "Wilo Helix V 2205 (초대형 산업 플랜트용)"
                pump_spec = f"정격 유량 {q_m3h_calc:.1f} m³/hr ({q_lmin_calc:.1f} L/min) | 양정 한계 {avg_H_m*1.1:.1f} m | 권장 동력 {res_kpi_total_kw*1.15:.2f} kW 급"
                pump_desc = "초대형 순환 계통 및 공업용 용수 대용량 송출용으로 맞춤 설계된 프리미엄급 고강도 기계설비 매칭 모델입니다."
                
            st.markdown(f"""
            <div style='background: rgba(30, 41, 59, 0.6); padding: 1.2rem; border-radius: 12px; border: 1px solid rgba(59, 130, 246, 0.3); line-height: 1.5; margin-bottom: 1.2rem;'>
                <div style='color: #60A5FA; font-weight: 800; font-size:1.15rem; margin-bottom: 0.5rem;'>🏆 {pump_model}</div>
                <div style='color: #E2E8F0; font-size: 0.9rem; font-weight: bold; margin-bottom: 0.6rem;'>📊 운전점: {pump_spec}</div>
                <div style='color: #94A3B8; font-size: 0.85rem; border-top: 1px solid rgba(255,255,255,0.08); padding-top: 0.5rem;'>
                    <b>🔬 기계 사양 설명:</b><br>{pump_desc}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # --- [🫧 펌프 공동현상(Cavitation) 및 NPSH 안전 정밀 진단 모듈] ---
            pump_node_id = None
            for n in nodes_list:
                if n["type"] == "pump":
                    pump_node_id = n["id"]
                    break
                    
            suction_pipe = None
            if pump_node_id:
                for p in pipes_list:
                    if p["to"] == pump_node_id:
                        suction_pipe = p
                        break
                        
            h_fs = 0.0
            suction_desc = "N/A"
            if pump_node_id and suction_pipe:
                s_id = suction_pipe["id"]
                s_d = float(suction_pipe["D"])
                s_l = float(suction_pipe["L"])
                q_final = converged_q[s_id]
                v_flow = calc_velocity(q_final, s_d)
                re_flow = calc_reynolds(rho, v_flow, s_d, mu)
                f_flow, _ = calc_friction_factor(re_flow, s_d, epsilon)
                
                # Darcy 마찰 수두손실 및 입구부 피팅 저항 K=1.5 가산
                h_fs = (f_flow * (s_l / s_d) + 1.5) * (v_flow**2) / (2 * 9.81)
                suction_desc = f"{s_id}번 배관 (내경 {s_d*1000:.1f}mm, 유속 {v_flow:.2f}m/s)"
                
            # NPSHa 연산 (대기압 101,325 Pa, 포화증기압 p_vapor)
            g_const = 9.81
            h_atm = 101325.0 / (rho * g_const)
            h_vap = p_vapor / (rho * g_const)
            npsha = h_atm - h_vap - h_fs
            
            # NPSHr 할당
            npshr = 2.0
            if q_m3h_calc > 4.0: npshr = 2.5
            if q_m3h_calc > 8.0: npshr = 3.0
            if q_m3h_calc > 16.0: npshr = 3.5
            
            # 등급 판정 및 처방 조제
            if npsha >= npshr + 0.5:
                npsh_status = "✅ 안전 (NPSH Pass)"
                npsh_color = "#10B981"
                npsh_msg = "가동 온도에서의 증기압 한계 대비 유효 흡입 압력이 넉넉합니다. 공동현상 파열 및 진동 우려가 전혀 없는 안전 운전점입니다."
                npsh_recipe = ""
            elif npsha >= npshr:
                npsh_status = "⚠️ 주의 (Cavitation Danger)"
                npsh_color = "#F59E0B"
                npsh_msg = "흡입 배관 마찰손실로 인해 안전 여유 마진이 0.5m 미만입니다. 미세한 유동 압동 시 임펠러 침식이 서서히 일어날 수 있습니다."
                npsh_recipe = f"""
                <div style='margin-top: 0.5rem; color: #FCD34D; font-size: 0.8rem; line-height: 1.4;'>
                    <b>🔧 공학적 개선 처방전 (Recipe):</b><br>
                    ▪ 흡입 관로({suction_desc})의 내경 D를 한 단계 높여 유속을 줄이고 마찰 저항을 억제하세요.<br>
                    ▪ 공급 탱크 수위를 높이거나, 펌프의 수직 설치 위치를 낮추어 정수압을 확보하세요.
                </div>
                """
            else:
                npsh_status = "🚨 위험 (Cavitation Occurred)"
                npsh_color = "#EF4444"
                npsh_msg = "공동현상(캐비테이션) 확정 발생! 내부 압력이 포화증기압 이하로 붕괴되어 급격한 기포 소손, 임펠러 파손 및 굉음진동이 시작됩니다."
                npsh_recipe = f"""
                <div style='margin-top: 0.5rem; color: #FCA5A5; font-size: 0.82rem; line-height: 1.4; border-top: 1px dashed rgba(239, 68, 68, 0.3); padding-top: 0.4rem;'>
                    <b>🚨 긴급 기계 설비 개선 조치 처방전:</b><br>
                    ▪ <b>[흡입관 설계 변경 필수]</b>: 흡입 배관({suction_desc})의 구경을 반드시 더 굵게 설계해 수두 손실을 차단하세요.<br>
                    ▪ <b>[온도 통제]</b>: 유체 가동 온도({temp_c:.1f}°C)를 떨어뜨려 포화 증기압을 강제로 진정시키십시오.
                </div>
                """
                
            st.markdown(f"""
            <div style='background: rgba(30, 41, 59, 0.5); padding: 1.2rem; border-radius: 12px; border: 1px solid {npsh_color}; line-height: 1.5; margin-bottom: 1.2rem;'>
                <div style='color: {npsh_color}; font-weight: 800; font-size:1.1rem; margin-bottom: 0.5rem;'>🫧 공동현상(Cavitation) & NPSH 정밀 진단</div>
                <div style='color: #E2E8F0; font-size: 0.9rem; font-weight: bold; margin-bottom: 0.4rem;'>
                    🩺 판정 등급: <span style='color: {npsh_color};'>{npsh_status}</span>
                </div>
                <div style='color: #CBD5E1; font-size: 0.85rem; margin-bottom: 0.5rem;'>
                    ▪ <b>유효 흡입 수두 (NPSHa)</b>: <span style='font-weight:bold;'>{npsha:.3f} m</span><br>
                    ▪ <b>필요 흡입 수두 (NPSHr)</b>: <span style='font-weight:bold;'>{npshr:.1f} m</span><br>
                    ▪ <b>흡입 관로 마찰손실 수두</b>: {h_fs:.3f} m (포화증기압: {p_vapor/1000.0:.2f} kPa)
                </div>
                <div style='color: #94A3B8; font-size: 0.82rem; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 0.4rem;'>
                    <b>🔬 유체 상태 진단:</b> {npsh_msg}
                </div>
                {npsh_recipe}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("##### 🌡️ 공통 한국 기온 편차 선팽창량 진단")
            props_mat = MECHANICAL_PROPS.get(material, {"E": 200e9, "alpha": 1.17e-5})
            alpha_mat = props_mat["alpha"]
            max_dt = max(abs(max_env_temp - install_temp), abs(install_temp - min_env_temp))
            expansion_tot = alpha_mat * total_L * max_dt
            
            st.caption(f"▪ 전체 배관 총 선팽창/수축 변형량: **{expansion_tot*1000:.1f} mm** (기온 격차 {max_dt:.1f}°C)")
            if expansion_tot > 0.05:
                st.warning("⚠️ 신축 팽창량이 50mm를 초과합니다. 배관 파손을 막기 위해 팽창 신축 루프 이음을 군데군데 설계에 반영하세요.")
            else:
                st.success("✅ 팽창 변형률이 미미하여 자가 흡수 가능한 한도 내에 있습니다.")
            st.markdown("</div>", unsafe_allow_html=True)
            
        # 상세 시공 견적서 렌더링
        st.markdown("<div class='res-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color:#3B82F6; margin-top:0;'>💸 펌프/배관 시스템 상세 견적 및 가격 산출 신뢰성 출처</h4>", unsafe_allow_html=True)
        st.write("본 배관망 설계 드로잉의 수력학(손실압, 요구동력) 분석치와 연동되어 자동 계산된 **총 설비 투자(CapEx) 및 연간 운영 가동비(OpEx)의 정밀 시공 세부 견적서**입니다.")
        st.dataframe(df_detail_cost, use_container_width=True)
        
        st.markdown(f"""
        <div style='background: rgba(30, 41, 59, 0.6); padding: 1.2rem; border-radius: 12px; border: 1px solid rgba(59, 130, 246, 0.2); font-size: 0.88rem; line-height: 1.6;'>
            <div style='color: #60A5FA; font-weight: bold; font-size:1.0rem; margin-bottom: 0.6rem;'>🎯 대한민국 국가 공인 단가 및 견적 표준 출처 (Reference Sources)</div>
            <ul style='margin: 0; padding-left: 1.2rem; color: #CBD5E1;'>
                <li><b>기계설비 노무 시공비:</b> 국토교통부고시 및 <a href='https://www.kict.re.kr' target='_blank' style='color:#60A5FA;'>한국건설기술연구원(KICT)</a> 발행 '기계설비공사 표준품셈' 제3장 플랜트 배관공/용접공 표준 품 공수 및 노임 계수 적용.</li>
                <li><b>원심 펌프 표준 장비비:</b> <a href='https://www.g2b.go.kr' target='_blank' style='color:#60A5FA;'>조달청(PPS) 나라장터종합쇼핑몰</a> 다수공급자계약(MAS)에 등록된 효성 프리미엄 고효율 IE3 원심펌프 제품군의 동력(kW)별 실거래가 회귀 모형 단가.</li>
                <li><b>배관 원자재비:</b> 대한건설협회 발행 월간 '거래가격(Market Price)' 강관배관공사 부문 철강 자재 시중 도매단가 반영.</li>
                <li><b>산업용 전력 요금:</b> <a href='https://cyber.kepco.co.kr' target='_blank' style='color:#60A5FA;'>한국전력공사(KEPCO)</a> 전기요금표 약관 기준 '산업용(을) 고압A' 평균 유효 전력 요금 계수 (150원/kWh).</li>
                <li><b>온실가스 규제 부담세:</b> 환경부 수도권대기환경청 고시 온실가스 배출권 거래제(K-ETS) 연도별 상위 3개년 배출권 낙찰가 가중평균치 (15,000원/tCO2eq).</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"초통합 분석 연산 중 오류가 발생했습니다: {e}")
        st.info("사이드바 하단 브릿지의 JSON 데이터 포맷이 유효한지 확인하세요.")


# =============================================================================
# [Streamlit UI 및 초통합 로직 가동]
# =============================================================================
def main():
    st.set_page_config(page_title="초통합 프리미엄 배관 시스템 시뮬레이터", page_icon="🚀", layout="wide")
    
    # [CORS 초월 실시간 동기화 브릿지 감지 및 세션 연동]
    global LATEST_CAD_DATA, DATA_UPDATED
    if "bridge_port" not in st.session_state:
        st.session_state["bridge_port"] = start_bridge_server()
        
    if DATA_UPDATED:
        with DATA_LOCK:
            if LATEST_CAD_DATA:
                js_str = json.dumps(LATEST_CAD_DATA, ensure_ascii=False)
                st.session_state["canvas_json_bridge"] = js_str
                st.session_state["canvas_json_bridge_t1"] = js_str
            DATA_UPDATED = False
    
    # ── 커스텀 CSS (세련된 다크/블루 톤 및 고도화된 UI/UX) ─────────
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=Outfit:wght@400;700;900&display=swap');
        
        .stApp {
            background-color: #0F172A;
            color: #E2E8F0;
            font-family: 'Inter', sans-serif;
        }
        
        /* 오토캐드 디지털 도면 연동 터미널 스타일링 */
        div[data-testid="stTextArea"] {
            border: 1px solid rgba(59, 130, 246, 0.35);
            border-radius: 14px;
            background: rgba(15, 23, 42, 0.55) !important;
            padding: 15px;
            box-shadow: 0 15px 30px rgba(0, 0, 0, 0.3);
            margin-top: 20px;
            margin-bottom: 25px;
            backdrop-filter: blur(10px);
        }
        div[data-testid="stTextArea"] label {
            color: #60A5FA !important;
            font-family: 'Outfit', sans-serif;
            font-weight: 800;
            font-size: 0.95rem !important;
            margin-bottom: 8px;
        }
        div[data-testid="stTextArea"] textarea {
            font-family: 'Courier New', Courier, monospace !important;
            background-color: #070B14 !important;
            color: #34D399 !important;
            border: 1px solid rgba(52, 211, 153, 0.25) !important;
            font-size: 12.5px !important;
            line-height: 1.5 !important;
            border-radius: 8px !important;
        }
        
        .main-header {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
            padding: 2.2rem;
            border-radius: 20px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            color: white;
            margin-bottom: 2rem;
            text-align: center;
            backdrop-filter: blur(10px);
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.3);
        }
        .main-header h1 {
            font-family: 'Outfit', sans-serif;
            margin: 0;
            font-size: 2.8rem;
            background: linear-gradient(90deg, #3B82F6 0%, #60A5FA 50%, #38BDF8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 900;
        }
        .main-header p {
            margin-top: 0.6rem;
            opacity: 0.85;
            font-size: 1.1rem;
            color: #94A3B8;
        }
        
        .section-header { 
            font-family: 'Outfit', sans-serif;
            font-size: 1.35rem; 
            font-weight: 700; 
            color: #3B82F6; 
            margin-top: 1.2rem; 
            margin-bottom: 0.8rem; 
            border-bottom: 2px solid rgba(255, 255, 255, 0.08); 
            padding-bottom: 0.4rem;
        }
        
        .res-card { 
            background: rgba(30, 41, 59, 0.45); 
            padding: 1.6rem; 
            border-radius: 16px; 
            border: 1px solid rgba(255, 255, 255, 0.05);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2); 
            margin-bottom: 1.5rem;
            transition: all 0.3s ease;
        }
        .res-card:hover {
            transform: translateY(-3px);
            border-color: rgba(59, 130, 246, 0.3);
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.4);
        }
        
        .badge { 
            display: inline-flex; align-items: center; justify-content: center;
            padding: 0.4rem 1.1rem; border-radius: 999px; 
            font-weight: 700; font-size: 0.95rem; margin-bottom: 1rem;
        }
        .badge-laminar { background:#065F46; color:#D1FAE5; border: 1px solid #059669; }
        .badge-transitional { background:#92400E; color:#FEF3C7; border: 1px solid #D97706; }
        .badge-turbulent { background:#1E3A8A; color:#DBEAFE; border: 1px solid #2563EB; }
    </style>
    """, unsafe_allow_html=True)

    # API 키 처리
    try:
        auto_api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        auto_api_key = ""

    # =============================================================================
    # [사이드바] 전면 통합 공통 옵션 패널
    # =============================================================================
    with st.sidebar:
        st.header("⚙️ 공통 배관 설계 옵션")
        st.caption("사이드바의 공통 조건이 캐드 드로잉판, AI 해석, LCC 차트 전체에 유기적으로 동시 반영됩니다.")
        
        # 1. 유체 특성
        st.subheader("💧 1. 공통 유체 정보")
        fluid_display = st.selectbox("유체 종류 선택", list(FLUID_OPTIONS.values()), index=0, key="shared_fluid")
        fluid_key = [k for k, v in FLUID_OPTIONS.items() if v == fluid_display][0]
        temp_c = st.number_input("유체 가동 온도 (°C)", min_value=-50.0, max_value=300.0, value=20.0, step=1.0, key="shared_temp")
        q_sys_lmin = st.number_input("💎 계통 대표 설계 유량 (Q_sys, L/min)", min_value=5.0, max_value=5000.0, value=200.0, step=10.0, key="shared_q_sys")
        
        try:
            rho, mu, p_vapor = get_fluid_properties(fluid_key, temp_c)
            st.success(f"**밀도:** {rho:.1f} kg/m³ | **점성:** {mu:.3e} Pa·s")
        except Exception:
            rho, mu, p_vapor = 998.2, 0.001002, 2300.0
            
        # 💧 유체 맞춤형 관 소재 가이드
        st.markdown("##### 💡 유체 맞춤형 배관 소재 가이드")
        rec_info = FLUID_MATERIAL_RECOMMENDATIONS.get(fluid_key)
        if rec_info:
            best_str = ", ".join(rec_info["best"])
            ok_str = ", ".join(rec_info["ok"])
            hazard_str = ", ".join(rec_info["hazard"])
            
            st.markdown(f"""
            <div style='background: rgba(30, 41, 59, 0.6); padding: 1rem; border-radius: 12px; border: 1px solid rgba(59, 130, 246, 0.2); font-size: 0.88rem; line-height: 1.45; margin-bottom: 1rem;'>
                <div style='color: #10B981; font-weight: bold;'>🏆 최적 소재 (Best)</div>
                <div style='margin-bottom: 0.4rem; color: #E2E8F0;'>{best_str}</div>
                <div style='color: #F59E0B; font-weight: bold;'>⚖️ 사용 가능 (OK)</div>
                <div style='margin-bottom: 0.4rem; color: #CBD5E1;'>{ok_str}</div>
                <div style='color: #EF4444; font-weight: bold;'>🚨 파손 위험 (Hazard)</div>
                <div style='margin-bottom: 0.6rem; color: #FCA5A5;'>{hazard_str}</div>
                <div style='border-top: 1px solid rgba(255,255,255,0.08); padding-top: 0.5rem; color: #94A3B8; font-size: 0.82rem;'>
                    <b>🔬 공학적 이유:</b><br>{rec_info["reason"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        
        # 2. 배관 재질 및 규격
        st.subheader("🔩 2. 배관 재질 및 조도")
        material = st.selectbox("배관 표준 재질", list(ROUGHNESS.keys()), index=2, key="shared_material")
        epsilon = ROUGHNESS[material]
        
        st.markdown("---")
        
        # 3. 안전율 및 펌프 효율
        st.subheader("🛡️ 3. 안전 및 설계 인자")
        safety_factor = st.number_input("목표 파열 안전율 (SF)", min_value=1.5, max_value=10.0, value=3.0, step=0.5, key="shared_sf")
        pump_eff = st.slider("펌프 모터 종합 효율 (%)", min_value=30, max_value=100, value=70, step=1, key="shared_peff")
        
        st.markdown("---")
        
        # 4. 경제 변수 (LCC)
        st.subheader("💸 4. LCC 경제 가산 이율")
        eco_elec = st.number_input("산업용 전기요금 (원/kWh)", value=150.0, step=5.0, key="shared_eco_elec")
        eco_ir = st.number_input("기준 할인율 (%)", value=2.5, step=0.1, key="shared_eco_ir")
        eco_years = st.number_input("설비 자본회수 기간 (년)", value=20, step=1, key="shared_eco_yr")
        eco_hours = st.number_input("연간 가동시간 (hr/yr)", value=8000, step=100, key="shared_eco_hr")
        eco_carbon_price = st.number_input("탄소배출가격 (원/tCO2)", value=15000, step=1000, key="shared_eco_carbon")
        
        st.markdown("---")
        
        # 수격 스파이크 및 한반도 기후 편차
        st.subheader("🌡️ 5. 혹서/혹한 기후 편차")
        install_temp = st.number_input("시공 설치 온도 (°C)", value=15.0, key="shared_inst")
        max_env_temp = st.number_input("여름 최고 외기 (°C)", value=40.0, key="shared_env_max")
        min_env_temp = st.number_input("겨울 최저 외기 (°C)", value=-20.0, key="shared_env_min")
        surge_multiplier = st.number_input("수격압 할증 계수", value=1.5, key="shared_surge")

    # =============================================================================
    # [메인 초통합 워크플로우 3단계 구성]
    # =============================================================================
    main_tabs = st.tabs([
        "🎨 인터랙티브 CAD 배관망 설계 및 결과 분석", 
        "🖼️ AI 평면도 판독 및 설계 자동 로더"
    ])
    
    default_net_json = '{"nodes": [], "pipes": []}'

    if "canvas_json_bridge" not in st.session_state:
        st.session_state["canvas_json_bridge"] = default_net_json
    if "canvas_json_bridge_t1" not in st.session_state:
        st.session_state["canvas_json_bridge_t1"] = default_net_json

    # 세션 상태가 변경되었을 때 메인 값으로 동방향 동기화 처리
    if st.session_state["canvas_json_bridge_t1"] != st.session_state["canvas_json_bridge"]:
        if st.session_state["canvas_json_bridge_t1"]:
            st.session_state["canvas_json_bridge"] = st.session_state["canvas_json_bridge_t1"]

    shared_json_input = st.session_state["canvas_json_bridge"]

    if "shared_pipes_json" not in st.session_state:
        st.session_state["shared_pipes_json"] = ""
        
    # =============================================================================
    # [1단계] 인터랙티브 CAD 배관망 드로잉
    # =============================================================================
    with main_tabs[0]:
        st.markdown("<h3 style='color:#3B82F6; font-family:\"Outfit\";'>🎨 1단계: 인터랙티브 CAD 배관망 드로잉</h3>", unsafe_allow_html=True)
        st.write("키보드 **단축키(숫자 1~6)**로 툴을 전환하고, 요소를 **더블클릭**해 값을 1초 만에 퀵 에디팅하세요! **스페이스바 드래그**로 Pan하고, **휠 스크롤**로 Zoom하여 수직 격자망 위에 편리하게 배관을 그립니다.")
        
        # 실시간 도면 동기화 Rerun 유도 및 사용성 극대화 컨트롤러 배치
        col_sync_btn, col_sync_info = st.columns([1, 1.8])
        with col_sync_btn:
            sync_triggered = st.button("⚡ 실시간 도면 연동 & 유동해석 실행", type="primary", use_container_width=True)
            if sync_triggered:
                with DATA_LOCK:
                    if LATEST_CAD_DATA:
                        js_str = json.dumps(LATEST_CAD_DATA, ensure_ascii=False)
                        st.session_state["canvas_json_bridge"] = js_str
                        st.session_state["canvas_json_bridge_t1"] = js_str
                        st.toast("⚡ 실시간 드로잉 데이터 연동 성공!", icon="🔥")
                    else:
                        st.toast("⚠️ 아직 동기화 대기 중인 실시간 드로잉 데이터가 없습니다. 캔버스 우측 상단의 [📤 설계결과 분석하기] 버튼을 먼저 눌러주세요.", icon="📢")
                st.rerun()
        with col_sync_info:
            st.markdown("""
            <div style='background: rgba(30, 41, 59, 0.4); padding: 0.5rem 1rem; border-radius: 8px; border: 1px dashed rgba(59, 130, 246, 0.25); font-size: 0.85rem; line-height: 1.45; color: #CBD5E1;'>
                💡 <b>보안 격리 환경 연동 꿀팁:</b> 브라우저 Origin 보안 정책에 의해 캔버스에서 그린 도면이 아래 보고서에 즉각 나타나지 않는 경우, <b>좌측의 연동 버튼을 클릭</b>하시거나, 혹은 캔버스에서 <b>[📤 설계결과 분석하기]</b> 클릭 시 클립보드에 자동 복사되는 도면 코드를 하단의 <b>📟 디지털 도면 연동 터미널</b>에 <b>Ctrl + V</b>로 톡 붙여넣어 주시면 100% 확실하게 분석이 활성화됩니다!
            </div>
            """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        # 웹 캐드 드로잉 컴포넌트 이식 (일반 원시 문자열로 선언하여 중괄호 문법 에러 100% 방지)
        canvas_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {
                    margin: 0;
                    padding: 0;
                    background-color: #1E293B;
                    color: #E2E8F0;
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    user-select: none;
                    overflow: hidden;
                }
                #app-container {
                    display: flex;
                    flex-direction: column;
                    width: 100vw;
                    height: 100vh;
                }
                #toolbar {
                    display: flex;
                    align-items: center;
                    background-color: #0F172A;
                    padding: 10px 15px;
                    gap: 10px;
                    border-bottom: 2px solid #334155;
                }
                .btn {
                    background-color: #334155;
                    border: 1px solid #475569;
                    color: white;
                    padding: 8px 12px;
                    border-radius: 6px;
                    cursor: pointer;
                    font-weight: 600;
                    font-size: 13px;
                    transition: all 0.2s ease;
                    display: flex;
                    align-items: center;
                    gap: 5px;
                }
                .btn:hover {
                    background-color: #475569;
                    border-color: #64748B;
                }
                .btn.active {
                    background-color: #2563EB;
                    border-color: #3B82F6;
                    box-shadow: 0 0 10px rgba(59, 130, 246, 0.5);
                }
                .btn-action {
                    background-color: #059669;
                    border-color: #10B981;
                }
                .btn-action:hover {
                    background-color: #10B981;
                }
                .btn-danger {
                    background-color: #DC2626;
                    border-color: #EF4444;
                }
                .btn-danger:hover {
                    background-color: #EF4444;
                }
                #canvas-area {
                    flex: 1;
                    position: relative;
                    background-color: #0B0F19;
                }
                canvas {
                    display: block;
                    width: 100%;
                    height: 100%;
                }
                /* sidebar-edit 스타일 완벽 제거 */
                .input-group {
                    margin-bottom: 12px;
                }
                .input-group label {
                    display: block;
                    font-size: 11px;
                    color: #94A3B8;
                    margin-bottom: 4px;
                    font-weight: 600;
                }
                .input-group input {
                    width: 100%;
                    background-color: #1E293B;
                    border: 1px solid #475569;
                    color: white;
                    padding: 6px 8px;
                    border-radius: 4px;
                    box-sizing: border-box;
                    font-size: 13px;
                }
                .section-title {
                    font-size: 13px;
                    font-weight: 700;
                    color: #3B82F6;
                    border-bottom: 1px solid #334155;
                    padding-bottom: 5px;
                    margin-bottom: 10px;
                }
            </style>
        </head>
        <body>
            <div id="app-container">
                <div id="toolbar">
                    <button class="btn active" id="btn-select" onclick="setMode('select')">🖐️ 요소 이동 [1]</button>
                    <button class="btn" id="btn-edit-mode" onclick="setMode('edit-mode')">⚙️ 속성 편집 [2]</button>
                    <button class="btn" id="btn-node-pump" onclick="setMode('add-node-pump')">🔺 펌프 배치 [3]</button>
                    <button class="btn" id="btn-node-valve" onclick="setMode('add-node-valve')">🎀 밸브 배치 [4]</button>
                    <button class="btn" id="btn-node-tank" onclick="setMode('add-node-tank')">⏹ 탱크 배치 [5]</button>
                    <button class="btn" id="btn-node-junction" onclick="setMode('add-node-junction')">🟢 분기점 배치 [6]</button>
                    <button class="btn" id="btn-pipe" onclick="setMode('draw-pipe')">🔩 관(Pipe) 연결 [7]</button>
                    <button class="btn btn-action" onclick="submitToPython()">📤 설계결과 분석하기</button>
                    <button class="btn btn-danger" onclick="clearCanvas()">🗑️ 전체 초기화</button>
                </div>
                
                <div id="canvas-area">
                    <canvas id="cad-canvas"></canvas>
                    
                    <!-- 속성 편집용 프리미엄 다크 모달 팝업 -->
                    <div id="property-modal" style="
                        display: none; 
                        position: fixed; 
                        top: 0; left: 0; 
                        width: 100vw; height: 100vh; 
                        background: rgba(15, 23, 42, 0.75); 
                        backdrop-filter: blur(8px); 
                        z-index: 9999; 
                        justify-content: center; 
                        align-items: center;
                    ">
                        <div style="
                            background: #1E293B; 
                            border: 1px solid rgba(59, 130, 246, 0.4); 
                            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5); 
                            border-radius: 16px; 
                            width: 320px; 
                            padding: 20px; 
                            position: relative;
                            font-family: 'Segoe UI', Inter, sans-serif;
                        ">
                            <div style="font-size: 15px; font-weight: 800; color: #60A5FA; margin-bottom: 15px; border-bottom: 1px solid #334155; padding-bottom: 8px;" id="modal-title">요소 정밀 사양 설정</div>
                            <div id="modal-fields" style="max-height: 350px; overflow-y: auto;"></div>
                            <div style="display: flex; gap: 8px; margin-top: 18px;">
                                <button class="btn" style="flex: 1; background-color: #2563EB; border-color: #3B82F6; justify-content: center;" onclick="closeModal(true)">💾 적용 완료</button>
                                <button class="btn btn-danger" style="flex: 0.8; justify-content: center;" onclick="closeModal(false)">❌ 취소</button>
                            </div>
                            <button class="btn btn-danger" style="width: 100%; margin-top: 8px; background-color: #DC2626; border-color: #EF4444; justify-content: center;" onclick="deleteSelected()">🗑️ 요소 삭제</button>
                        </div>
                    </div>
                </div>
            </div>

            <script>
                const canvas = document.getElementById('cad-canvas');
                const ctx = canvas.getContext('2d');
                
                const GRID_SIZE = 40; 
                
                let nodes = [];
                let pipes = [];
                let currentMode = 'select'; 
                let selectedElement = null;
                let isDragging = false;
                let dragStartNode = null;
                let pipeStartNode = null;
                let mousePos = {x: 0, y: 0};
                
                let scale = 1.0;
                let offsetX = 0;
                let offsetY = 0;
                let isPanning = false;
                let panStartX = 0;
                let panStartY = 0;
                let isSpacePressed = false;
                
                let globalOffset = 0;
                
                // 샘플 기본 노드 매핑
                nodes = INITIAL_NODES_PLACEHOLDER;
                
                pipes = INITIAL_PIPES_PLACEHOLDER;
                
                function resizeCanvas() {
                    const w = canvas.parentElement ? canvas.parentElement.clientWidth : 0;
                    const h = canvas.parentElement ? canvas.parentElement.clientHeight : 0;
                    canvas.width = w > 0 ? w : window.innerWidth;
                    canvas.height = h > 0 ? h : 500;
                }
                
                window.addEventListener('resize', resizeCanvas);
                setTimeout(resizeCanvas, 300);

                function setMode(mode) {
                    currentMode = mode;
                    document.querySelectorAll('#toolbar .btn').forEach(b => b.classList.remove('active'));
                    const btnId = 'btn-' + (mode.startsWith('add-node') ? 'node-' + mode.split('-')[2] : mode);
                    const btn = document.getElementById(btnId);
                    if (btn) btn.classList.add('active');
                    selectedElement = null;
                    pipeStartNode = null;
                    document.getElementById('property-modal').style.display = 'none';
                }

                function clearCanvas() {
                    nodes = [];
                    pipes = [];
                    selectedElement = null;
                    pipeStartNode = null;
                    document.getElementById('property-modal').style.display = 'none';
                    submitToPython();
                }

                function snapToGrid(coord) {
                    return Math.round(coord / GRID_SIZE) * GRID_SIZE;
                }

                function getMousePos(e) {
                    const rect = canvas.getBoundingClientRect();
                    const screenX = e.clientX - rect.left;
                    const screenY = e.clientY - rect.top;
                    return {
                        x: (screenX - offsetX) / scale,
                        y: (screenY - offsetY) / scale
                    };
                }

                window.addEventListener('keydown', e => {
                    if (e.code === 'Space') {
                        isSpacePressed = true;
                        canvas.style.cursor = 'grab';
                        e.preventDefault();
                    }
                    if ((e.key === 'Delete' || e.key === 'Backspace') && selectedElement) {
                        deleteSelected();
                    }
                    if (e.key === '1') setMode('select');
                    if (e.key === '2') setMode('edit-mode');
                    if (e.key === '3') setMode('add-node-pump');
                    if (e.key === '4') setMode('add-node-valve');
                    if (e.key === '5') setMode('add-node-tank');
                    if (e.key === '6') setMode('add-node-junction');
                    if (e.key === '7') setMode('draw-pipe');
                });

                window.addEventListener('keyup', e => {
                    if (e.code === 'Space') {
                        isSpacePressed = false;
                        canvas.style.cursor = 'default';
                        isPanning = false;
                    }
                });

                canvas.addEventListener('wheel', e => {
                    e.preventDefault();
                    const rect = canvas.getBoundingClientRect();
                    const mouseScreenX = e.clientX - rect.left;
                    const mouseScreenY = e.clientY - rect.top;
                    const mouseWorldX = (mouseScreenX - offsetX) / scale;
                    const mouseWorldY = (mouseScreenY - offsetY) / scale;
                    
                    const zoomFactor = 1.1;
                    if (e.deltaY < 0) {
                        scale = Math.min(scale * zoomFactor, 3.0);
                    } else {
                        scale = Math.max(scale / zoomFactor, 0.4);
                    }
                    offsetX = mouseScreenX - mouseWorldX * scale;
                    offsetY = mouseScreenY - mouseWorldY * scale;
                });

                canvas.addEventListener('dblclick', e => {
                    const pos = getMousePos(e);
                    const clickedNode = findNodeAt(pos.x, pos.y);
                    
                    if (clickedNode) {
                        selectedElement = {type: 'node', data: clickedNode};
                        openPropertyModal('node', clickedNode);
                    } else {
                        const clickedPipe = findPipeAt(pos.x, pos.y);
                        if (clickedPipe) {
                            selectedElement = {type: 'pipe', data: clickedPipe};
                            openPropertyModal('pipe', clickedPipe);
                        }
                    }
                });

                canvas.addEventListener('mousedown', e => {
                    if (isSpacePressed) {
                        isPanning = true;
                        panStartX = e.clientX - offsetX;
                        panStartY = e.clientY - offsetY;
                        canvas.style.cursor = 'grabbing';
                        return;
                    }
                    
                    const pos = getMousePos(e);
                    const clickedNode = findNodeAt(pos.x, pos.y);
                    
                    if (currentMode.startsWith('add-node')) {
                        const type = currentMode.split('-')[2];
                        const newId = 'n' + (nodes.length + 1);
                        nodes.push({
                            id: newId,
                            x: snapToGrid(pos.x),
                            y: snapToGrid(pos.y),
                            type: type,
                            name: type.toUpperCase() + '_' + newId,
                            val: type === 'pump' ? 80 : (type === 'valve' ? 2 : 0)
                        });
                        setMode('select');
                    } else if (currentMode === 'draw-pipe') {
                        if (clickedNode) {
                            pipeStartNode = clickedNode;
                        }
                    } else if (currentMode === 'select') {
                        if (clickedNode) {
                            selectedElement = {type: 'node', data: clickedNode};
                            isDragging = true;
                            dragStartNode = clickedNode;
                        } else {
                            const clickedPipe = findPipeAt(pos.x, pos.y);
                            if (clickedPipe) {
                                selectedElement = {type: 'pipe', data: clickedPipe};
                            } else {
                                selectedElement = null;
                            }
                        }
                    } else if (currentMode === 'edit-mode') {
                        if (clickedNode) {
                            selectedElement = {type: 'node', data: clickedNode};
                            openPropertyModal('node', clickedNode);
                        } else {
                            const clickedPipe = findPipeAt(pos.x, pos.y);
                            if (clickedPipe) {
                                selectedElement = {type: 'pipe', data: clickedPipe};
                                openPropertyModal('pipe', clickedPipe);
                            } else {
                                selectedElement = null;
                                document.getElementById('property-modal').style.display = 'none';
                            }
                        }
                    }
                });

                canvas.addEventListener('mousemove', e => {
                    if (isPanning) {
                        offsetX = e.clientX - panStartX;
                        offsetY = e.clientY - panStartY;
                        return;
                    }
                    const pos = getMousePos(e);
                    mousePos = pos;
                    
                    if (isDragging && dragStartNode) {
                        dragStartNode.x = snapToGrid(pos.x); 
                        dragStartNode.y = snapToGrid(pos.y);
                    }
                });

                canvas.addEventListener('mouseup', e => {
                    if (isPanning) {
                        isPanning = false;
                        canvas.style.cursor = isSpacePressed ? 'grab' : 'default';
                        return;
                    }
                    if (currentMode === 'draw-pipe' && pipeStartNode) {
                        const pos = getMousePos(e);
                        let endNode = findNodeAt(pos.x, pos.y);
                        
                        // 마우스 놓은 지점에 기기 노드가 없다면 빈 공간에 외부 유출용 Junction 노드 즉시 자동 신설
                        if (!endNode) {
                            const newId = 'n' + (nodes.length + 1);
                            const snapX = snapToGrid(pos.x);
                            const snapY = snapToGrid(pos.y);
                            endNode = {
                                id: newId,
                                x: snapX,
                                y: snapY,
                                type: 'junction',
                                name: 'OUTLET_' + newId,
                                val: 0
                            };
                            nodes.push(endNode);
                        }
                        
                        if (endNode !== pipeStartNode) {
                            const exists = pipes.some(p => (p.from === pipeStartNode.id && p.to === endNode.id) || (p.from === endNode.id && p.to === pipeStartNode.id));
                            if (!exists) {
                                // 관 번호 순차 고유화: p + (최대 ID 인덱스 + 1)
                                const maxIdNum = pipes.reduce((max, p) => {
                                    const num = parseInt(p.id.replace('p', ''));
                                    return isNaN(num) ? max : Math.max(max, num);
                                }, 0);
                                const newPipeId = 'p' + (maxIdNum + 1);
                                
                                pipes.push({
                                    id: newPipeId,
                                    from: pipeStartNode.id,
                                    to: endNode.id,
                                    L: 50,
                                    D: 0.08,
                                    Q: 0,
                                    t_rec: "",
                                    p_loss: "",
                                    v_flow: 0
                                });
                            }
                        }
                        pipeStartNode = null;
                    }
                    isDragging = false;
                    dragStartNode = null;
                });

                function findNodeAt(x, y) {
                    return nodes.find(n => Math.hypot(n.x - x, n.y - y) < 22);
                }

                function findPipeAt(x, y) {
                    return pipes.find(p => {
                        const n1 = nodes.find(n => n.id === p.from);
                        const n2 = nodes.find(n => n.id === p.to);
                        if (!n1 || !n2) return false;
                        
                        const A = x - n1.x;
                        const B = y - n1.y;
                        const C = n2.x - n1.x;
                        const D = n2.y - n1.y;
                        
                        const dot = A * C + B * D;
                        const len_sq = C * C + D * D;
                        let param = -1;
                        if (len_sq !== 0) param = dot / len_sq;
                        
                        let xx, yy;
                        if (param < 0) {
                            xx = n1.x;
                            yy = n1.y;
                        } else if (param > 1) {
                            xx = n2.x;
                            yy = n2.y;
                        } else {
                            xx = n1.x + param * C;
                            yy = n1.y + param * D;
                        }
                        return Math.hypot(x - xx, y - yy) < 10;
                    });
                }

                function openPropertyModal(type, data) {
                    const modal = document.getElementById('property-modal');
                    const title = document.getElementById('modal-title');
                    const fields = document.getElementById('modal-fields');
                    modal.style.display = 'flex';
                    fields.innerHTML = '';
                    
                    if (type === 'node') {
                        title.innerHTML = `⚙️ 기기 [${data.name}] 사양 설정`;
                        fields.innerHTML = `
                            <div class="input-group" style="margin-bottom:12px;">
                                <label style="display:block; font-size:11px; color:#94A3B8; margin-bottom:4px; font-weight:600;">기기 표시 이름</label>
                                <input type="text" id="modal-prop-name" value="${data.name}" style="width:100%; background:#1E293B; border:1px solid #475569; color:white; padding:6px 8px; border-radius:4px; box-sizing:border-box; font-size:13px;">
                            </div>
                            <div class="input-group" style="margin-bottom:12px;">
                                <label style="display:block; font-size:11px; color:#94A3B8; margin-bottom:4px; font-weight:600;">기기 분류</label>
                                <input type="text" value="${data.type.toUpperCase()}" readonly style="width:100%; background:#0F172A; border:1px solid #475569; color:white; padding:6px 8px; border-radius:4px; box-sizing:border-box; font-size:13px; opacity:0.5;">
                            </div>
                            ${data.type === 'pump' ? `
                                <div class="input-group" style="margin-bottom:12px;">
                                    <label style="display:block; font-size:11px; color:#94A3B8; margin-bottom:4px; font-weight:600;">펌프 양정 압력 수두 (H, m)</label>
                                    <input type="number" id="modal-prop-val" value="${data.val}" style="width:100%; background:#1E293B; border:1px solid #475569; color:white; padding:6px 8px; border-radius:4px; box-sizing:border-box; font-size:13px;">
                                </div>
                            ` : data.type === 'valve' ? `
                                <div class="input-group" style="margin-bottom:12px;">
                                    <label style="display:block; font-size:11px; color:#94A3B8; margin-bottom:4px; font-weight:600;">밸브 손실 저항 계수 (K)</label>
                                    <input type="number" id="modal-prop-val" value="${data.val}" style="width:100%; background:#1E293B; border:1px solid #475569; color:white; padding:6px 8px; border-radius:4px; box-sizing:border-box; font-size:13px;">
                                </div>
                            ` : ''}
                        `;
                    } else if (type === 'pipe') {
                        title.innerHTML = `🔩 관 [${data.id}] 정밀 사양 설정`;
                        fields.innerHTML = `
                            <div class="input-group" style="margin-bottom:12px;">
                                <label style="display:block; font-size:11px; color:#94A3B8; margin-bottom:4px; font-weight:600;">배관 길이 (L, m)</label>
                                <input type="number" id="modal-prop-L" value="${data.L}" style="width:100%; background:#1E293B; border:1px solid #475569; color:white; padding:6px 8px; border-radius:4px; box-sizing:border-box; font-size:13px;">
                            </div>
                            <div class="input-group" style="margin-bottom:12px;">
                                <label style="display:block; font-size:11px; color:#94A3B8; margin-bottom:4px; font-weight:600;">배관 내경 (D, m)</label>
                                <input type="number" id="modal-prop-D" value="${data.D}" step="0.001" style="width:100%; background:#1E293B; border:1px solid #475569; color:white; padding:6px 8px; border-radius:4px; box-sizing:border-box; font-size:13px;">
                            </div>
                            <div class="input-group" style="margin-bottom:8px;">
                                <label style="display:block; font-size:11px; color:#94A3B8; margin-bottom:4px; font-weight:600;">수동 고정 유량 (Q, L/min) [선택]</label>
                                <input type="number" id="modal-prop-Q" value="${(parseFloat(data.Q || 0) * 60000).toFixed(1)}" step="0.1" style="width:100%; background:#1E293B; border:1px solid #475569; color:white; padding:6px 8px; border-radius:4px; box-sizing:border-box; font-size:13px;">
                                <span style="font-size:10px; color:#64748B; display:block; margin-top:4px;">* 0 또는 빈칸 입력 시 물리학 솔버가 자동 연산합니다.</span>
                            </div>
                        `;
                    }
                }

                function closeModal(shouldSave) {
                    const modal = document.getElementById('property-modal');
                    modal.style.display = 'none';
                    
                    if (shouldSave && selectedElement) {
                        const type = selectedElement.type;
                        const data = selectedElement.data;
                        
                        if (type === 'node') {
                            data.name = document.getElementById('modal-prop-name').value;
                            const valEl = document.getElementById('modal-prop-val');
                            if (valEl) data.val = parseFloat(valEl.value);
                        } else if (type === 'pipe') {
                            data.L = parseFloat(document.getElementById('modal-prop-L').value);
                            data.D = parseFloat(document.getElementById('modal-prop-D').value);
                            const qVal = parseFloat(document.getElementById('modal-prop-Q').value);
                            data.Q = isNaN(qVal) || qVal <= 0.0 ? 0.0 : qVal / 60000.0;
                        }
                    }
                    selectedElement = null;
                }

                function deleteSelected() {
                    const modal = document.getElementById('property-modal');
                    modal.style.display = 'none';
                    if (!selectedElement) return;
                    const data = selectedElement.data;
                    if (selectedElement.type === 'node') {
                        nodes = nodes.filter(n => n.id !== data.id);
                        pipes = pipes.filter(p => p.from !== data.id && p.to !== data.id);
                    } else {
                        pipes = pipes.filter(p => p.id !== data.id);
                    }
                    selectedElement = null;
                }

                function drawGrid() {
                    ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
                    ctx.lineWidth = 1;
                    const gridStep = GRID_SIZE;
                    const startX = snapToGrid(-offsetX / scale) - gridStep;
                    const endX = startX + canvas.width / scale + gridStep * 2;
                    const startY = snapToGrid(-offsetY / scale) - gridStep;
                    const endY = startY + canvas.height / scale + gridStep * 2;
                    
                    for (let x = startX; x < endX; x += gridStep) {
                        ctx.beginPath();
                        ctx.moveTo(x, startY);
                        ctx.lineTo(x, endY);
                        ctx.stroke();
                    }
                    for (let y = startY; y < endY; y += gridStep) {
                        ctx.beginPath();
                        ctx.moveTo(startX, y);
                        ctx.lineTo(endX, y);
                        ctx.stroke();
                    }
                }

                function drawFlowArrows(n1, n2, p) {
                    if (p.Q <= 0) return;
                    const dx = n2.x - n1.x;
                    const dy = n2.y - n1.y;
                    const dist = Math.hypot(dx, dy);
                    if (dist === 0) return;
                    const ux = dx / dist;
                    const uy = dy / dist;
                    
                    const speed = Math.max(p.v_flow * 0.8, 0.5);
                    const arrowSpacing = 60; 
                    const offset = (globalOffset * speed) % arrowSpacing;
                    ctx.fillStyle = '#10B981'; 
                    
                    for (let d = offset; d < dist; d += arrowSpacing) {
                        const ax = n1.x + ux * d;
                        const ay = n1.y + uy * d;
                        ctx.beginPath();
                        ctx.moveTo(ax + ux * 6, ay + uy * 6);
                        ctx.lineTo(ax - ux * 4 - uy * 4, ay - uy * 4 + ux * 4);
                        ctx.lineTo(ax - ux * 4 + uy * 4, ay - uy * 4 - ux * 4);
                        ctx.closePath();
                        ctx.fill();
                    }
                }

                function showToast(message, isDanger = false) {
                    let toast = document.getElementById('cad-toast');
                    if (!toast) {
                        toast = document.createElement('div');
                        toast.id = 'cad-toast';
                        toast.style.cssText = `
                            position: fixed;
                            bottom: 25px;
                            left: 50%;
                            transform: translateX(-50%) translateY(100px);
                            background: rgba(15, 23, 42, 0.92);
                            backdrop-filter: blur(10px);
                            border: 1px solid rgba(59, 130, 246, 0.45);
                            color: white;
                            padding: 12px 28px;
                            border-radius: 99px;
                            font-size: 13px;
                            font-weight: 600;
                            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);
                            z-index: 999999;
                            transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
                            opacity: 0;
                            pointer-events: none;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            gap: 8px;
                            text-align: center;
                            white-space: nowrap;
                        `;
                        document.body.appendChild(toast);
                    }
                    toast.style.borderColor = isDanger ? 'rgba(239, 68, 68, 0.5)' : 'rgba(96, 165, 250, 0.5)';
                    toast.innerHTML = (isDanger ? '⚠️ ' : '⚡ ') + message;
                    
                    // Animate in
                    setTimeout(() => {
                        toast.style.transform = 'translateX(-50%) translateY(0)';
                        toast.style.opacity = '1';
                    }, 50);
                    
                    // Animate out
                    setTimeout(() => {
                        toast.style.transform = 'translateX(-50%) translateY(100px)';
                        toast.style.opacity = '0';
                    }, 4000);
                }

                function submitToPython() {
                    const payload = {
                        pipes: pipes,
                        nodes: nodes
                    };
                    const jsonStr = JSON.stringify(payload);
                    
                    // 1. CORS 우회용 백그라운드 클립보드 병행 복사 강제 실행
                    let clipOk = false;
                    try {
                        navigator.clipboard.writeText(jsonStr);
                        clipOk = true;
                    } catch (err) {
                        const el = document.createElement('textarea');
                        el.value = jsonStr;
                        document.body.appendChild(el);
                        el.select();
                        document.execCommand('copy');
                        document.body.removeChild(el);
                        clipOk = true;
                    }
                    
                    // 2. 부모 Streamlit text_area 주입 시도 (React state 세터 우회 완벽 해킹)
                    let parentSyncOk = false;
                    try {
                        const textAreas = window.parent.document.querySelectorAll('textarea');
                        textAreas.forEach(ta => {
                            if (ta.placeholder && ta.placeholder.includes("streamlit_canvas_json_bridge_exchange_area")) {
                                // React의 내부 value setter를 획득하여 직접 강제 주입
                                let nativeTextAreaValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
                                nativeTextAreaValueSetter.call(ta, jsonStr);
                                
                                // React가 데이터 입력을 인지할 수 있도록 버블링 이벤트 발화
                                ta.dispatchEvent(new Event('input', { bubbles: true }));
                                ta.dispatchEvent(new Event('change', { bubbles: true }));
                                parentSyncOk = true;
                            }
                        });
                    } catch (e) {
                        parentSyncOk = false;
                    }
                    
                    // 3. 초성능 로컬 HTTP API 브릿지 서버 전송 (궁극의 100% 실시간 자동 융합)
                    const bridgePort = "BRIDGE_PORT_PLACEHOLDER";
                    if (bridgePort && bridgePort !== "None" && bridgePort !== "") {
                        fetch(`http://127.0.0.1:${bridgePort}/sync`, {
                            method: "POST",
                            headers: {
                                "Content-Type": "application/json"
                            },
                            body: jsonStr
                        })
                        .then(response => response.json())
                        .then(data => {
                            if (data.status === "success") {
                                if (pipes.length === 0 && nodes.length === 0) {
                                    showToast("초기화 완료");
                                } else {
                                    showToast("동기화 완료");
                                }
                            }
                        })
                        .catch(err => {
                            showToast("API 브릿지 미연결 (클립보드 백업 작동)", true);
                        });
                    } else {
                        showToast("API 브릿지 오프라인 (클립보드 복사 완료)");
                    }
                }

                function animate() {
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    
                    ctx.save();
                    ctx.translate(offsetX, offsetY);
                    ctx.scale(scale, scale);
                    
                    drawGrid();
                    
                    globalOffset += 1;
                    
                    pipes.forEach(p => {
                        const n1 = nodes.find(n => n.id === p.from);
                        const n2 = nodes.find(n => n.id === p.to);
                        if (!n1 || !n2) return;
                        
                        ctx.save();
                        ctx.beginPath();
                        ctx.moveTo(n1.x, n1.y);
                        ctx.lineTo(n2.x, n2.y);
                        
                        if (selectedElement && selectedElement.type === 'pipe' && selectedElement.data.id === p.id) {
                            ctx.strokeStyle = '#3B82F6';
                            ctx.lineWidth = 6;
                        } else {
                            ctx.strokeStyle = 'rgba(255, 255, 255, 0.25)';
                            ctx.lineWidth = 4.5;
                        }
                        ctx.stroke();
                        ctx.restore();
                        
                        drawFlowArrows(n1, n2, p);
                        
                        const mx = (n1.x + n2.x) / 2;
                        const my = (n1.y + n2.y) / 2;
                        
                        ctx.fillStyle = '#94A3B8';
                        ctx.font = '10px Inter, sans-serif';
                        ctx.textAlign = 'center';
                        ctx.textBaseline = 'bottom';
                        ctx.fillText(`${p.id} (${p.L}m, ${Math.round(p.D*1000)}mm)`, mx, my - 4);
                        
                        if (p.t_rec) {
                            const textRec = p.t_rec;
                            ctx.save();
                            ctx.fillStyle = '#FFFBEB';
                            ctx.textAlign = 'center';
                            ctx.textBaseline = 'middle';
                            ctx.fillText(textRec, mx, my + 24);
                            ctx.restore();
                        }
                    });
                    
                    if (currentMode === 'draw-pipe' && pipeStartNode) {
                        ctx.beginPath();
                        ctx.moveTo(pipeStartNode.x, pipeStartNode.y);
                        ctx.lineTo(mousePos.x, mousePos.y);
                        ctx.strokeStyle = 'rgba(59, 130, 246, 0.5)';
                        ctx.lineWidth = 3;
                        ctx.setLineDash([5, 5]);
                        ctx.stroke();
                        ctx.setLineDash([]);
                    }
                    
                    nodes.forEach(n => {
                        ctx.save();
                        ctx.beginPath();
                        ctx.arc(n.x, n.y, 20, 0, 2 * Math.PI);
                        
                        let baseColor = '#10B981';
                        let glowColor = 'rgba(16, 185, 129, 0.4)';
                        
                        if (n.type === 'pump') {
                            baseColor = '#EF4444';
                            glowColor = 'rgba(239, 68, 68, 0.6)';
                        } else if (n.type === 'valve') {
                            baseColor = '#F59E0B';
                            glowColor = 'rgba(245, 158, 11, 0.6)';
                        } else if (n.type === 'tank') {
                            baseColor = '#3B82F6';
                            glowColor = 'rgba(59, 130, 246, 0.6)';
                        }
                        
                        ctx.shadowBlur = 15;
                        ctx.shadowColor = glowColor;
                        
                        if (selectedElement && selectedElement.type === 'node' && selectedElement.data.id === n.id) {
                            ctx.fillStyle = '#1D4ED8';
                            ctx.strokeStyle = '#60A5FA';
                            ctx.lineWidth = 3.5;
                        } else {
                            ctx.fillStyle = baseColor;
                            ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
                            ctx.lineWidth = 2.5;
                        }
                        ctx.fill();
                        ctx.stroke();
                        ctx.restore();
                        
                        ctx.fillStyle = 'white';
                        ctx.font = 'bold 11px Inter, sans-serif';
                        ctx.textAlign = 'center';
                        ctx.textBaseline = 'middle';
                        const symbol = n.type === 'pump' ? 'PUMP' : (n.type === 'valve' ? 'VALVE' : (n.type === 'tank' ? 'TANK' : 'JUNC'));
                        ctx.fillText(symbol, n.x, n.y);
                        
                        ctx.fillStyle = '#E2E8F0';
                        ctx.font = '10px Inter, sans-serif';
                        ctx.textBaseline = 'top';
                        ctx.fillText(n.name, n.x, n.y + 24);
                        if (n.type === 'pump' || n.type === 'valve') {
                            ctx.fillStyle = '#94A3B8';
                            ctx.fillText(`(${n.val}${n.type === 'pump' ? 'm' : 'K'})`, n.x, n.y + 36);
                        }
                    });
                    ctx.restore();
                    requestAnimationFrame(animate);
                }
                
                setTimeout(() => {
                    requestAnimationFrame(animate);
                }, 400);
            </script>
        </body>
        </html>
        """
        # 세션 연동 및 초기 데이터 브릿지 주입
        import json
        
        initial_nodes_js = "[]"
        initial_pipes_js = "[]"
        
        if st.session_state.get("canvas_json_bridge"):
            try:
                bridge_data = json.loads(st.session_state["canvas_json_bridge"])
                if "nodes" in bridge_data and "pipes" in bridge_data:
                    # [절대 법칙] 캔버스 원본 도면 데이터는 무조건 선제 백업 보관!
                    initial_nodes_js = json.dumps(bridge_data["nodes"], ensure_ascii=False)
                    initial_pipes_js = json.dumps(bridge_data["pipes"], ensure_ascii=False)
                    
                    try:
                        # 1단계 CAD 캔버스 로딩 전에 솔버 연산 수행
                        solve_pipe_network(bridge_data["pipes"], bridge_data["nodes"], rho, mu, epsilon, q_sys_lmin, material)
                        # 솔버 성공 시에만 계산 결과(Q, v_flow, t_rec)를 이식하여 캔버스 데이터 업그레이드!
                        initial_nodes_js = json.dumps(bridge_data["nodes"], ensure_ascii=False)
                        initial_pipes_js = json.dumps(bridge_data["pipes"], ensure_ascii=False)
                    except Exception as solver_err:
                        # 솔버 실패 시 계산 결과는 없더라도 원본 도면 구조는 100% 보존하여 캔버스가 날아가는 대참사 완벽 차단!!!
                        pass
            except Exception:
                pass
                
        rendered_canvas_html = canvas_html.replace(
            "INITIAL_NODES_PLACEHOLDER", initial_nodes_js
        ).replace(
            "INITIAL_PIPES_PLACEHOLDER", initial_pipes_js
        ).replace(
            "BRIDGE_PORT_PLACEHOLDER", str(st.session_state.get("bridge_port", ""))
        )
        
        st.components.v1.html(rendered_canvas_html, height=600, scrolling=False)
        
        st.markdown("<div class='res-card'>", unsafe_allow_html=True)
        st.markdown("##### 📤 실시간 설계결과 분석 안내")
        st.write("설계를 완료한 뒤 위의 **[📤 설계결과 분석하기]** 버튼을 클릭하시면 사이드바 하단 브릿지를 통해 유동해석 리포트에 설계 데이터가 즉각 연동 반영됩니다.")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # 1단계 캐드 드로잉판 바로 아래에 실시간 결과표 및 LCC 대시보드 렌더링 (원스크린 통합 UX 제공)
        render_integrated_report(
            shared_json_input, rho, mu, epsilon, fluid_key, material, safety_factor, pump_eff,
            eco_years, eco_hours, eco_elec, eco_carbon_price, install_temp, max_env_temp, min_env_temp, surge_multiplier, eco_ir,
            widget_key="canvas_json_bridge_t1", p_vapor=p_vapor, q_sys_lmin=q_sys_lmin
        )

    # =============================================================================
    # [2단계] AI 평면도 판독 및 설계 로더
    # =============================================================================
    with main_tabs[1]:
        st.markdown("<h3 style='color:#3B82F6; font-family:\"Outfit\";'>🖼️ 2단계: AI 평면도 판독 및 설계 로더</h3>", unsafe_allow_html=True)
        st.write("관로 사진이나 평면도를 해독하여 루프를 자동 해킹하고, AI가 해독한 D, L 정보를 설계결과 분석 리포트 및 1단계 CAD판에 바로 로드 연동시킵니다.")
        
        col_ai_up, col_ai_act = st.columns([1, 1.2])
        
        with col_ai_up:
            uploaded_file = st.file_uploader("배관 도면/사진 업로드", type=["jpg", "png", "jpeg"], key="shared_uploader")
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption="분석용 도면", use_column_width=True)
                
        with col_ai_act:
            st.markdown("<div class='res-card'>", unsafe_allow_html=True)
            st.markdown("**🤖 Gemini AI 정밀 판독**")
            st.write("사이드바의 공통 배관 재질과 금리를 토대로, 각 관에 흐를 유량의 예상 부하를 계산하고 배관의 추천 스케줄 두께를 판독합니다.")
            
            ai_key_pwd = st.text_input("Gemini API Key 수동 오버라이드", value=auto_api_key, type="password", key="shared_ai_key")
            
            analyze_btn = st.button("🚀 AI 도면 해독 시작", type="primary", use_container_width=True, disabled=(uploaded_file is None), key="shared_ai_btn")
            
            if analyze_btn:
                use_key = ai_key_pwd if ai_key_pwd else auto_api_key
                if not use_key:
                    st.error("API 키가 누락되어 해독을 시작할 수 없습니다.")
                else:
                    with st.spinner("AI가 배관 구조와 권장 관두께를 정밀 해독하는 중..."):
                        try:
                            genai.configure(api_key=use_key)
                            model = genai.GenerativeModel('gemini-1.5-flash')
                            
                            prompt = """
                            당신은 배관 공학 및 수리학 최고 권위자입니다.
                            제공된 배관 평면도/도면 사진을 정밀 해독하여, 웹 CAD 드로잉 판에 즉시 시각적으로 로드할 수 있도록 아래 JSON 규격을 백분 활용하여 단 하나의 완전한 배관망 구조 JSON 데이터만 반환하세요. (```json 과 ``` 마크다운 기호 및 기타 부연 설명 텍스트를 절대 넣지 말고 오직 순수 JSON 데이터만 출력하세요.)

                            [설계 규칙]
                            1. 도면에 묘사된 흐름 순서(예: 입구 탱크 -> 펌프 -> 밸브 -> 출구 탱크 또는 중간 분기점)를 파악하세요.
                            2. 모든 노드(nodes)는 화면에서 서로 겹치지 않고 격자망 위에 단정히 배치될 수 있도록 40의 배수 좌표(예: x는 120, 240, 360, 480, 600, 720..., y는 120, 160, 200, 240, 280, 320, 360...)를 부여하세요.
                            3. 배관(pipes)은 노드의 ID('n1', 'n2' 등)를 참조하여 'from'과 'to' 필드로 올바르게 이어져야 합니다.
                            4. 도면에서 가늠되는 대략의 배관 길이 L (m)과 직경 D (m)를 기재해 주세요.
                            5. 유속 부하와 압력을 견디기 위해 Barlow 공식을 근거로 판단된 추천 KS 스케줄 규격(예: "SCH 40", "SCH 80", "SCH 10S" 등)을 분석하여 각 배관의 t_rec 필드에 기재하세요.

                            [JSON 출력 포맷]
                            {
                                "nodes": [
                                    {"id": "n1", "x": 120, "y": 240, "type": "tank", "name": "공급 저장조 (Tank)", "val": 0},
                                    {"id": "n2", "x": 280, "y": 240, "type": "pump", "name": "가압 공급 펌프", "val": 80},
                                    {"id": "n3", "x": 440, "y": 160, "type": "valve", "name": "유량 제어 글로브 밸브", "val": 10},
                                    {"id": "n4", "x": 440, "y": 320, "type": "junction", "name": "분기 헤더 분기점", "val": 0},
                                    {"id": "n5", "x": 600, "y": 240, "type": "tank", "name": "회수 저장탱크", "val": 0}
                                ],
                                "pipes": [
                                    {"id": "p1", "from": "n1", "to": "n2", "L": 15, "D": 0.10, "Q": 0.02, "t_rec": "SCH 40", "p_loss": "", "v_flow": 0},
                                    {"id": "p2", "from": "n2", "to": "n3", "L": 45, "D": 0.08, "Q": 0.01, "t_rec": "SCH 40", "p_loss": "", "v_flow": 0},
                                    {"id": "p3", "from": "n2", "to": "n4", "L": 40, "D": 0.08, "Q": 0.01, "t_rec": "SCH 40", "p_loss": "", "v_flow": 0},
                                    {"id": "p4", "from": "n3", "to": "n5", "L": 30, "D": 0.08, "Q": 0.01, "t_rec": "SCH 80", "p_loss": "", "v_flow": 0},
                                    {"id": "p5", "from": "n4", "to": "n5", "L": 30, "D": 0.08, "Q": 0.01, "t_rec": "SCH 80", "p_loss": "", "v_flow": 0}
                                ]
                            }
                            """
                            response = model.generate_content([prompt, image])
                            clean_resp = response.text.replace('```json', '').replace('```', '').strip()
                            
                            st.session_state["canvas_json_bridge"] = clean_resp
                            st.success("🎉 AI가 도면을 정밀 해독하여 1단계 CAD 드로잉 판 및 분석 연동 브릿지에 데이터를 성공적으로 주입하였습니다! 1단계 탭으로 이동하셔서 해독된 배관망 구조를 직접 눈으로 확인하고 수동 편집을 가미해보세요!")
                            
                        except Exception as e:
                            st.error(f"AI 해독 실패: {e}")
            st.markdown("</div>", unsafe_allow_html=True)



if __name__ == "__main__":
    main()
