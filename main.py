from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from typing import List, Dict, Any

app = FastAPI(title="林梦龙时间序列预测API")

# 1. 配置 CORS (允许您的 GitHub Pages 域名访问)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 部署后请严格替换为您的 GitHub Pages 地址，例如 ["https://linmenglong.github.io"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. 鉴权配置
VALID_KEYS = ["linmenglong_vip_2024", "test_key_123"]
def verify_api_key(api_key: str):
    if api_key not in VALID_KEYS:
        raise HTTPException(status_code=403, detail="无效的 API 授权码，请联系作者购买！")
    return api_key

# 3. 请求数据模型
class ForecastRequest(BaseModel):
    data: List[float]
    horizon: int
    model: str
    params: Dict[str, Any]
    train_split: float = 0.8
    api_key: str

class TuneRequest(BaseModel):
    multi_data: List[List[float]]
    horizon: int
    target_metric: str
    train_split: float
    periods: List[int]
    api_key: str

# ================= 4. 核心算法移植 =================
def calc_metrics(actual, fit, train_size):
    # (翻译自你的 JS calculateValidationMetrics)
    n = 0; sum_abs = 0; sum_sq = 0; sum_abs_pct = 0; sum_sym_pct = 0; sum_act = 0
    for i in range(train_size, len(actual)):
        if fit[i] is None: continue
        y = actual[i]; f = fit[i]; err = y - f
        sum_abs += abs(err); sum_sq += err**2; sum_act += abs(y)
        if y != 0: sum_abs_pct += abs(err / y)
        denom = (abs(y) + abs(f)) / 2
        if denom != 0: sum_sym_pct += abs(err) / denom
        n += 1
    if n == 0: return {"mae": 0, "rmse": 0, "mape": 0, "mse": 0, "smape": 0, "wmape": 0}
    mae = sum_abs / n; mse = sum_sq / n; rmse = np.sqrt(mse)
    mape = (sum_abs_pct / n) * 100; smape = (sum_sym_pct / n) * 100
    wmape = (sum_abs / sum_act) * 100 if sum_act != 0 else 0
    return {"mae": mae, "rmse": rmse, "mape": mape, "mse": mse, "smape": smape, "wmape": wmape}

def forecast_algorithm(model_key, data, horizon, params):
    # (翻译自你的 JS forecastAlgorithm)
    n = len(data); fit = [None] * n; future = [0.0] * horizon
    if model_key == 'naive':
        for i in range(1, n): fit[i] = data[i-1]
        future = [data[-1]] * horizon
    elif model_key == 'sma':
        w = min(int(params.get('window', 3)), max(1, n-1))
        for i in range(w, n): fit[i] = sum(data[i-w:i]) / w
        future = [sum(data[-w:]) / w] * horizon
    elif model_key == 'ses':
        a = float(params.get('alpha', 0.2))
        fit[0] = data[0]
        for i in range(1, n): fit[i] = a * data[i-1] + (1 - a) * fit[i-1]
        future = [a * data[-1] + (1 - a) * fit[-1]] * horizon
    elif model_key == 'des':
        a = float(params.get('alpha', 0.2)); b = float(params.get('beta', 0.1))
        l = data[0]; t = data[1] - data[0] if n > 1 else 0; fit[0] = l
        for i in range(1, n):
            prev_l = l
            l = a * data[i] + (1 - a) * (l + t)
            t = b * (l - prev_l) + (1 - b) * t
            fit[i] = prev_l + t
        for h in range(1, horizon+1): future[h-1] = l + h * t
    elif model_key == 'tes':
        a = float(params.get('alpha', 0.2)); b = float(params.get('beta', 0.1)); g = float(params.get('gamma', 0.1))
        m = max(2, int(params.get('period', 12)))
        l = data[0]; t = 0; s = [1.0] * m
        for i in range(n):
            idx = i % m
            if i < m: fit[i] = data[i]
            else:
                prev_l = l
                l = a * (data[i] - s[idx]) + (1 - a) * (l + t)
                t = b * (l - prev_l) + (1 - b) * t
                s[idx] = g * (data[i] - l) + (1 - g) * s[idx]
                fit[i] = prev_l + t + s[idx]
        for h in range(1, horizon+1):
            idx = (n + h - 1) % m
            future[h-1] = l + h * t + s[idx]
    elif model_key.startswith('croston_'):
        a = float(params.get('alpha', 0.2)); b = float(params.get('beta', a))
        z = 0; p = 1; first_nonzero = False; q_counter = 0
        for i in range(n):
            q_counter += 1
            if data[i] != 0:
                if not first_nonzero: z = data[i]; p = q_counter; first_nonzero = True
                else:
                    z = a * data[i] + (1 - a) * z
                    p = (b * 1 + (1 - b) * p) if model_key == 'croston_tsb' else (a * q_counter + (1 - a) * p)
                q_counter = 0
            else:
                if model_key == 'croston_tsb': p = (1 - b) * p
            rate = 0
            if model_key == 'croston_orig': rate = z / max(1, p)
            elif model_key == 'croston_sba': rate = (1 - a/2) * (z / max(1, p))
            elif model_key == 'croston_sbj': rate = (1 - a/(2-a)) * (z / max(1, p))
            elif model_key == 'croston_tsb': rate = z * p
            fit[i] = rate
        future = [fit[-1] or 0] * horizon
    # 注：ARIMA/SARIMA 如要在后端实现，建议引入 statsmodels 库，这里暂略
    else:
        raise ValueError(f"模型 {model_key} 还未在 Python 后端实现")
    return fit, future

# ================= 5. API 路由 =================
@app.post("/api/forecast")
async def forecast(req: ForecastRequest, key: str = Depends(verify_api_key)):
    try:
        train_size = int(len(req.data) * req.train_split)
        fit, future = forecast_algorithm(req.model, req.data, req.horizon, req.params)
        metrics = calc_metrics(req.data, fit, train_size)
        return {"code": 200, "data": {"fit": fit, "future": future, "metrics": metrics}}
    except Exception as e:
        return {"code": 500, "message": str(e)}

@app.post("/api/tune")
async def tune(req: TuneRequest, key: str = Depends(verify_api_key)):
    # 简化版调优逻辑，遍历可能的参数组合
    models_to_test = ['naive', 'sma', 'ses', 'des', 'tes', 'croston_orig', 'croston_sba']
    best_model = "naive"; best_params = {}; min_metric = float('inf')
    # 这里仅演示逻辑，实际应用中应完善参数网格
    for m in models_to_test:
        # 模拟参数遍历
        grid = [{}] if m == 'naive' else [{"alpha": 0.5}, {"window": 3}]
        for p in grid:
            total_metric = 0
            for data in req.multi_data:
                train_size = int(len(data) * req.train_split)
                fit, _ = forecast_algorithm(m, data, req.horizon, p)
                metrics = calc_metrics(data, fit, train_size)
                total_metric += metrics.get(req.target_metric, 9999)
            avg_metric = total_metric / len(req.multi_data)
            if avg_metric < min_metric:
                min_metric = avg_metric; best_model = m; best_params = p
    return {"code": 200, "data": {"best_model": best_model, "best_params": best_params, "best_metric_val": min_metric}}