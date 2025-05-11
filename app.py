import streamlit as st
import pulp
import pandas as pd
import numpy as np
import altair as alt

# アプリのタイトル
st.title("在庫管理最適化問題")

# デフォルトのパラメータ値
default_h = 2.0      # 在庫保管コスト
default_p = 5.0      # 欠品コスト
default_K = 50       # 段取りコスト
default_initial_inventory = 10  # 初期在庫
default_M = 10000    # 十分大きな数

# デフォルトの月と需要の定義
default_months = ["1月", "2月", "3月", "4月", "5月", "6月", "7月", "8月", "9月", "10月", "11月", "12月"]
default_demands = [15, 10, 20, 10, 25, 20, 15, 15, 10, 20, 25, 15]

# パラメータ設定
st.header("パラメータ設定")

# コストパラメータをテーブルとして表示
st.subheader("コストパラメータ")

cost_params = pd.DataFrame({
    "パラメータ": ["在庫保管コスト (h)", "欠品コスト (p)", "段取りコスト (K)", "初期在庫量", "十分大きな数 (M)"],
    "値": [default_h, default_p, default_K, default_initial_inventory, default_M]
})

edited_cost_params = st.data_editor(
    cost_params,
    column_config={
        "パラメータ": st.column_config.TextColumn("パラメータ", disabled=True),
        "値": st.column_config.NumberColumn("値", min_value=0.0, step=0.1)
    },
    hide_index=True,
    use_container_width=True
)

# 編集されたパラメータを取得
h = edited_cost_params.loc[0, "値"]
p = edited_cost_params.loc[1, "値"]
K = edited_cost_params.loc[2, "値"]
initial_inventory = edited_cost_params.loc[3, "値"]
M = edited_cost_params.loc[4, "値"]

# 需要データの編集
st.subheader("月別需要")

# データフレームを作成
demands_data = pd.DataFrame({
    "月": default_months,
    "需要": default_demands
})

# データエディタを表示
edited_demands = st.data_editor(
    demands_data,
    num_rows="dynamic",  # 行の追加・削除を許可
    column_config={
        "月": st.column_config.TextColumn("月"),
        "需要": st.column_config.NumberColumn("需要", min_value=0, step=1)
    },
    use_container_width=True
)

# 編集されたデータを取得
months = edited_demands["月"].tolist()
demands = edited_demands["需要"].tolist()

# 最適化問題を解く
def solve_inventory_problem(h, p, K, initial_inventory, M, demands):
    T = len(demands)  # 期間数
    
    # 問題の定義
    prob = pulp.LpProblem("InventoryManagement", pulp.LpMinimize)
    
    # 決定変数
    x = {t: pulp.LpVariable(f"x_{t}", lowBound=0) for t in range(T)}
    I_plus = {t: pulp.LpVariable(f"I_plus_{t}", lowBound=0) for t in range(T)}
    I_minus = {t: pulp.LpVariable(f"I_minus_{t}", lowBound=0) for t in range(T)}
    y = {t: pulp.LpVariable(f"y_{t}", cat=pulp.LpBinary) for t in range(T)}
    
    # 初期条件
    I_plus_prev = initial_inventory
    I_minus_prev = 0
    
    # 目的関数
    prob += pulp.lpSum(h * I_plus[t] + p * I_minus[t] + K * y[t] for t in range(T))
    
    # 制約条件
    for t in range(T):
        # 在庫バランス制約
        prob += I_plus[t] - I_minus[t] == I_plus_prev - I_minus_prev + x[t] - demands[t]
        I_plus_prev, I_minus_prev = I_plus[t], I_minus[t]
        
        # 段取り判定用の制約
        prob += x[t] <= M * y[t]
    
    # 問題を解く
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # 結果を抽出
    status = pulp.LpStatus[prob.status]
    if status == 'Optimal':
        results = {
            "status": "最適解が見つかりました",
            "order_quantities": [pulp.value(x[t]) for t in range(T)],
            "positive_inventory": [pulp.value(I_plus[t]) for t in range(T)],
            "negative_inventory": [pulp.value(I_minus[t]) for t in range(T)],
            "setup_indicators": [pulp.value(y[t]) for t in range(T)],
            "objective_value": pulp.value(prob.objective)
        }
        
        # 派生した結果を計算
        results["net_inventory"] = [results["positive_inventory"][t] - results["negative_inventory"][t] for t in range(T)]
        results["holding_costs"] = [h * results["positive_inventory"][t] for t in range(T)]
        results["shortage_costs"] = [p * results["negative_inventory"][t] for t in range(T)]
        results["setup_costs"] = [K * results["setup_indicators"][t] for t in range(T)]
        results["total_costs"] = [results["holding_costs"][t] + results["shortage_costs"][t] + results["setup_costs"][t] for t in range(T)]
        
        return results
    else:
        return {"status": f"最適解が見つかりませんでした。ステータス: {status}"}

# 最適化の実行ボタン
if st.button("最適化を実行"):
    if len(demands) == 0:
        st.error("需要データが入力されていません。少なくとも1か月分のデータが必要です。")
    else:
        with st.spinner("計算中..."):
            results = solve_inventory_problem(h, p, K, initial_inventory, M, demands)
        
        if "status" in results and results["status"] == "最適解が見つかりました":
            # 結果を表示
            st.header("最適化結果")
            st.success(f"総コスト: {results['objective_value']:.2f}")
            
            # 結果をテーブルとして表示し、編集可能に
            results_data = pd.DataFrame({
                "月": months,
                "需要": demands,
                "発注量": [round(q, 1) for q in results["order_quantities"]],
                "発注の有無": [int(y) for y in results["setup_indicators"]],
                "正の在庫": [round(inv, 1) for inv in results["positive_inventory"]],
                "負の在庫 (欠品)": [round(neg, 1) for neg in results["negative_inventory"]],
                "正味在庫": [round(net, 1) for net in results["net_inventory"]],
                "在庫保管コスト": [round(hc, 1) for hc in results["holding_costs"]],
                "欠品コスト": [round(sc, 1) for sc in results["shortage_costs"]],
                "段取りコスト": [round(sc, 1) for sc in results["setup_costs"]],
                "合計コスト": [round(tc, 1) for tc in results["total_costs"]]
            })
            
            # 結果を表示 - ユーザが見るだけで編集はできない
            st.subheader("詳細結果")
            st.dataframe(results_data, use_container_width=True)
            
            # 最適結果をCSV形式でダウンロードできるようにする
            csv = results_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="結果をCSVでダウンロード",
                data=csv,
                file_name='inventory_optimization_results.csv',
                mime='text/csv',
            )
            
            # グラフで可視化
            st.subheader("月ごとの在庫・需要・発注の推移")
            
            # データフレームを準備
            chart_data = pd.DataFrame({
                '月': months,
                '需要': demands,
                '発注量': results["order_quantities"],
                '正味在庫': results["net_inventory"]
            })
            
            # 長形式に変換
            chart_data_long = pd.melt(chart_data, id_vars=['月'], value_vars=['需要', '発注量', '正味在庫'],
                                  var_name='項目', value_name='値')
            
            # Altairでグラフ作成
            chart = alt.Chart(chart_data_long).mark_line(point=True).encode(
                x=alt.X('月', sort=None),
                y='値',
                color='項目',
                tooltip=['月', '項目', '値']
            ).properties(
                width=700,
                height=400
            ).interactive()
            
            st.altair_chart(chart, use_container_width=True)
            
            # コスト内訳のグラフ
            st.subheader("コスト内訳")
            
            cost_data = pd.DataFrame({
                '月': months,
                '在庫保管コスト': results["holding_costs"],
                '欠品コスト': results["shortage_costs"],
                '段取りコスト': results["setup_costs"]
            })
            
            cost_data_long = pd.melt(cost_data, id_vars=['月'], value_vars=['在庫保管コスト', '欠品コスト', '段取りコスト'],
                                  var_name='コスト種類', value_name='コスト')
            
            cost_chart = alt.Chart(cost_data_long).mark_bar().encode(
                x=alt.X('月', sort=None),
                y='コスト',
                color='コスト種類',
                tooltip=['月', 'コスト種類', 'コスト']
            ).properties(
                width=700,
                height=400
            )
            
            st.altair_chart(cost_chart, use_container_width=True)
            
            # 累積コストのグラフ
            st.subheader("累積コスト推移")
            cumulative_costs = np.cumsum(results["total_costs"])
            
            cumulative_data = pd.DataFrame({
                '月': months,
                '累積コスト': cumulative_costs
            })
            
            cumulative_chart = alt.Chart(cumulative_data).mark_line(point=True).encode(
                x=alt.X('月', sort=None),
                y='累積コスト',
                tooltip=['月', '累積コスト']
            ).properties(
                width=700,
                height=400
            )
            
            st.altair_chart(cumulative_chart, use_container_width=True)
            
        else:
            st.error(results.get("status", "エラーが発生しました"))
else:
    # 初回表示時の説明
    st.info("「最適化を実行」ボタンをクリックして計算を開始してください。")