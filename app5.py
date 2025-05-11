import streamlit as st
import pulp
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from sklearn.manifold import MDS
import io

st.title("工場内物流最適化問題（配送経路問題）")

# デフォルトの距離行列
default_distance_matrix = np.array([
    [0, 10, 15, 20, 10, 25],
    [10, 0, 35, 25, 15, 20],
    [15, 35, 0, 30, 20, 25],
    [20, 25, 30, 0, 15, 10],
    [10, 15, 20, 15, 0, 10],
    [25, 20, 25, 10, 10, 0]
])

# 距離行列の編集機能
st.subheader("ステーション間の距離行列")

# 距離行列のサイズを選択できる機能（デフォルトは6x6）
n_stations = st.number_input("ステーション数", min_value=3, max_value=10, value=6)

# 既存の距離行列をリサイズ
if n_stations != len(default_distance_matrix):
    if n_stations < len(default_distance_matrix):
        # 小さくする場合は切り詰め
        resized_matrix = default_distance_matrix[:n_stations, :n_stations]
    else:
        # 大きくする場合は新しい要素を0で埋める
        resized_matrix = np.zeros((n_stations, n_stations))
        resized_matrix[:len(default_distance_matrix), :len(default_distance_matrix)] = default_distance_matrix
    
    # 対角成分は0に設定
    for i in range(n_stations):
        resized_matrix[i, i] = 0
        
    distance_matrix_init = resized_matrix
else:
    distance_matrix_init = default_distance_matrix

# 距離行列をデータフレームに変換
distance_df = pd.DataFrame(
    distance_matrix_init,
    columns=[f"ステーション {i+1}" for i in range(n_stations)],
    index=[f"ステーション {i+1}" for i in range(n_stations)]
)

# 距離行列を編集可能なテーブルとして表示
st.write("ステーション間の距離を編集できます（対角成分は常に0）:")
edited_distance_df = st.data_editor(
    distance_df,
    use_container_width=True,
    height=400
)

# CSVでのアップロードとダウンロード機能
col1, col2 = st.columns(2)

with col1:
    # 現在の距離行列をCSVでダウンロード
    csv_buffer = io.StringIO()
    edited_distance_df.to_csv(csv_buffer)
    st.download_button(
        label="距離行列をCSVでダウンロード",
        data=csv_buffer.getvalue(),
        file_name="distance_matrix.csv",
        mime="text/csv"
    )

with col2:
    # CSVファイルをアップロードして距離行列を更新
    uploaded_file = st.file_uploader("距離行列をCSVからアップロード", type="csv")
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file, index_col=0)
            if uploaded_df.shape[0] != uploaded_df.shape[1]:
                st.error("アップロードされたCSVは正方行列ではありません")
            else:
                # アップロードされた行列のサイズを確認
                new_n = uploaded_df.shape[0]
                if 3 <= new_n <= 10:
                    st.success(f"{new_n}x{new_n} の距離行列をアップロードしました")
                    # 行列を更新（再度ページをリロードする必要あり）
                    st.experimental_rerun()
                else:
                    st.error("距離行列のサイズは3から10の間である必要があります")
        except Exception as e:
            st.error(f"CSVの読み込みエラー: {e}")

# 編集された距離行列をnumpy配列に変換
distance_matrix = edited_distance_df.values

# 編集後の行列が対称行列になるように補正
n = len(distance_matrix)
for i in range(n):
    distance_matrix[i, i] = 0  # 対角成分は常に0
    for j in range(i+1, n):
        # i→jとj→iの距離の平均を取る
        avg_dist = (distance_matrix[i, j] + distance_matrix[j, i]) / 2
        distance_matrix[i, j] = avg_dist
        distance_matrix[j, i] = avg_dist

# 編集された距離行列を表示
st.subheader("現在の距離行列")
st.write("（対称行列に補正されています）")
st.table(pd.DataFrame(
    distance_matrix,
    columns=[f"ステーション {i+1}" for i in range(n)],
    index=[f"ステーション {i+1}" for i in range(n)]
))

# 距離のディクショナリを作成
distances = {}
for i in range(1, n+1):
    for j in range(1, n+1):
        if i != j:
            distances[(i, j)] = distance_matrix[i-1][j-1]

# MDSを用いて2次元埋め込み（距離行列にできるだけ沿う配置）
try:
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    pos_mds = mds.fit_transform(distance_matrix)
    
    # ノード番号（1始まり）と2次元座標の辞書作成
    pos_dict = {i + 1: pos_mds[i] for i in range(n)}
    
    # 入力グラフの表示
    st.subheader("入力ネットワーク")
    
    # 完全グラフ（全てのエッジを持つグラフ）の作成
    fig_input, ax_input = plt.subplots(figsize=(10, 8))
    G_input = nx.Graph()
    nodes = list(range(1, n + 1))
    G_input.add_nodes_from(nodes)
    
    # 各ノード間のエッジを追加（重みとして距離を設定）
    for i in range(n):
        for j in range(i + 1, n):
            G_input.add_edge(i + 1, j + 1, weight=distance_matrix[i][j])
    
    # グラフ描画
    nx.draw(G_input, pos=pos_dict, with_labels=True, node_color='skyblue', 
            node_size=700, font_weight='bold')
    edge_labels = nx.get_edge_attributes(G_input, 'weight')
    nx.draw_networkx_edge_labels(G_input, pos=pos_dict, edge_labels=edge_labels)
    ax_input.set_title("Complete graphs based on distance matrices")
    plt.axis('off')
    plt.tight_layout()
    
    st.pyplot(fig_input)
    
    # PuLPで問題を解く
    def solve_tsp():
        # 問題の定義
        prob = pulp.LpProblem("AGV_TSP", pulp.LpMinimize)
        
        # 決定変数
        x = {}
        for i in range(1, n+1):
            for j in range(1, n+1):
                if i != j:
                    x[(i, j)] = pulp.LpVariable(f"x_{i}_{j}", cat=pulp.LpBinary)
        
        u = {}
        for i in range(1, n+1):
            u[i] = pulp.LpVariable(f"u_{i}", lowBound=1, upBound=n, cat=pulp.LpInteger)
        
        # 目的関数
        prob += pulp.lpSum(distances[(i, j)] * x[(i, j)] for i in range(1, n+1) for j in range(1, n+1) if i != j)
        
        # 制約条件
        # 各ステーションには1回だけ入る
        for j in range(1, n+1):
            prob += pulp.lpSum(x[(i, j)] for i in range(1, n+1) if i != j) == 1
        
        # 各ステーションからは1回だけ出る
        for i in range(1, n+1):
            prob += pulp.lpSum(x[(i, j)] for j in range(1, n+1) if i != j) == 1
        
        # Miller-Tucker-Zemlin制約（部分巡回路の除去）
        for i in range(2, n+1):
            for j in range(2, n+1):
                if i != j:
                    prob += u[i] - u[j] + n * x[(i, j)] <= n - 1
        
        # 問題を解く
        status = prob.solve())
        
        if status == pulp.LpStatusOptimal:
            # 結果の抽出
            route = []
            current = 1  # スタートは1
            route.append(current)
            
            for _ in range(n-1):
                next_node = None
                for j in range(1, n+1):
                    if j != current and pulp.value(x[(current, j)]) > 0.5:  # バイナリ変数の場合、数値誤差を考慮
                        next_node = j
                        break
                
                if next_node is None:
                    st.error("巡回路の抽出中にエラーが発生しました。")
                    return [], 0
                
                route.append(next_node)
                current = next_node
            
            route.append(1)  # 最後に1に戻る
            
            total_distance = sum(distances[(route[i-1], route[i])] for i in range(1, len(route)))
            
            return route, total_distance
        else:
            st.error(f"最適解が見つかりませんでした。ステータス: {pulp.LpStatus[status]}")
            return [], 0
    
    # 問題を解くボタン
    if st.button("最適巡回路を計算"):
        with st.spinner("計算中..."):
            route, total_distance = solve_tsp()
        
        if route:
            # 結果を表示
            st.subheader("最適解")
            st.success(f"最適巡回路: {' → '.join(map(str, route))}")
            st.success(f"総移動距離: {total_distance:.2f}")
            
            # 巡回路の各ステップを表として表示
            route_data = []
            for i in range(len(route)-1):
                route_data.append({
                    "ステップ": i+1,
                    "出発地": f"ステーション {route[i]}",
                    "目的地": f"ステーション {route[i+1]}",
                    "距離": distances[(route[i], route[i+1])]
                })
            
            route_df = pd.DataFrame(route_data)
            st.subheader("巡回路の詳細")
            st.table(route_df)
            
            # グラフ描画
            st.subheader("最適巡回路のグラフ表示")
            
            fig, ax = plt.subplots(figsize=(10, 8))
            G = nx.DiGraph()
            
            # ノードを追加
            for i in range(1, n+1):
                G.add_node(i)
            
            # エッジを追加
            for i in range(len(route)-1):
                G.add_edge(route[i], route[i+1])
            
            # ノードを描画
            nx.draw_networkx_nodes(G, pos_dict, node_size=700, node_color='lightblue')
            
            # エッジを描画
            edge_labels = {(route[i], route[i+1]): f"{i+1}" for i in range(len(route)-1)}
            nx.draw_networkx_edges(G, pos_dict, width=1.5, alpha=0.7, 
                                edge_color='blue', connectionstyle='arc3,rad=0.1',
                                arrowsize=15)
            
            # ラベルを描画
            nx.draw_networkx_labels(G, pos_dict)
            
            # エッジラベルを描画
            nx.draw_networkx_edge_labels(G, pos_dict, edge_labels=edge_labels, 
                                      font_color='red', font_weight='bold')
            
            plt.axis('off')
            plt.tight_layout()
            
            st.pyplot(fig)
    else:
        st.info("「最適巡回路を計算」ボタンをクリックして計算を開始してください。")
except Exception as e:
    st.error(f"計算中にエラーが発生しました: {e}")
    st.warning("距離行列の値を確認して、再度試してください。")
