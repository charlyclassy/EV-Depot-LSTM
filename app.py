from __future__ import annotations

import os, json, hashlib, random, time
from pathlib import Path
from datetime import datetime

os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

st.set_page_config(page_title="Optivolt EV Depot Intelligence Portal", page_icon="⚡", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html,body,[data-testid="stAppViewContainer"]{font-family:'Inter',sans-serif!important;background:#F8FAFC!important;color:#1E293B!important}
[data-testid="stSidebar"]{background:#0F172A!important}
[data-testid="stSidebar"] *{color:#E2E8F0!important}
[data-testid="stMetric"]{background:#fff;border:1px solid #E2E8F0;border-radius:8px;padding:10px 14px}
[data-testid="stMetricLabel"] p{font-size:.82rem!important;white-space:normal!important}
.stButton>button{background:#2563EB!important;color:#fff!important;border-radius:6px!important;border:none!important;font-weight:600!important}
.badge{padding:4px 10px;border-radius:4px;font-size:12px;font-weight:700;display:inline-block}
.safe{background:#DCFCE7;color:#15803D}.warn{background:#FEF9C3;color:#A16207}.danger{background:#FEE2E2;color:#B91C1C}
.perf{background:#fff;border:1px solid #E2E8F0;border-radius:12px;padding:18px}
</style>
""", unsafe_allow_html=True)

ORG_NAME="Optivolt Solutions Limited"
REAL_DEPOTS=["Bexleyheath","Dartford","Islington","Mount Pleasant","Orpington","Premier Park","Whitechapel","Camden","Victoria"]
DEPOT_CAPACITY_KW={"Bexleyheath":17.0,"Dartford":45.0,"Islington":13.0,"Mount Pleasant":260.0,"Orpington":14.0,"Premier Park":70.0,"Whitechapel":55.0,"Camden":21.0,"Victoria":29.0}
WARNING_ALPHA=.80; SAMPLING_MINUTES=15; STEPS_PER_DAY=96
DATA_START=pd.Timestamp("2021-07-08"); TRAIN_END=pd.Timestamp("2022-04-01"); VALID_END=pd.Timestamp("2022-05-10"); TEST_END=pd.Timestamp("2022-07-26")
SEED=42; LOOKBACK=96; EPOCHS=40; BATCH=256
random.seed(SEED); np.random.seed(SEED); tf.keras.utils.set_random_seed(SEED)
try: tf.config.experimental.enable_op_determinism()
except Exception: pass

BUNDLED=[Path("data/processed/depot_demand.csv"),Path("UKPN_OptimisePrime_9_Depots_Processed.csv")]
FILES={
"Bexleyheath":Path("UKPN_OptimisePrime_Bexleyheath.csv"),"Camden":Path("UKPN_OptimisePrime_Camden.csv"),"Dartford":Path("UKPN_OptimisePrime_Dartford.csv"),
"Islington":Path("UKPN_OptimisePrime_Islington.csv"),"Mount Pleasant":Path("UKPN_OptimisePrime_Mount_Pleasant.csv"),"Orpington":Path("UKPN_OptimisePrime_Orpington.csv"),
"Premier Park":Path("UKPN_OptimisePrime_Premier_Park.csv"),"Victoria":Path("UKPN_OptimisePrime_Victoria.csv"),"Whitechapel":Path("UKPN_OptimisePrime_Whitechapel.csv")}
METRICS_FILE=Path("model_comparison_metrics.csv"); SELECT_FILE=Path("model_selection_decision.csv"); PREDS_FILE=Path("heldout_predictions_all_models.csv"); FUTURE_FILE=Path("lstm_24h_recursive_forecast.csv")
STATE=Path(".app_state"); UP_DIR=STATE/"uploaded_depots"; REG_FILE=STATE/"uploaded_depots_registry.json"; MODEL_DIR=STATE/"lstm_models"

@st.cache_data(show_spinner=False)
def load_real_depot_data(name):
    for p in BUNDLED:
        if p.exists():
            r=pd.read_csv(p,parse_dates=["timestamp"],dtype={"depot_id":str}); req={"timestamp","depot_id","demand_kw"}
            if not req.issubset(r.columns): raise ValueError(f"{p} is missing required columns")
            d=r.loc[r["depot_id"].astype(str)==name,["timestamp","demand_kw"]]
            if len(d): return d.rename(columns={"timestamp":"Timestamp","demand_kw":"Demand_kW"}).sort_values("Timestamp").reset_index(drop=True)
    p=FILES.get(name)
    if p is None or not p.exists(): raise FileNotFoundError(f"Data file for {name} was not found: {p}")
    r=pd.read_csv(p,parse_dates=["timestamp"])
    return r[["timestamp","demand_kw"]].rename(columns={"timestamp":"Timestamp","demand_kw":"Demand_kW"}).sort_values("Timestamp").reset_index(drop=True)

def standardise(df):
    d=df.copy(); d.columns=[str(c).strip() for c in d.columns]
    if "Timestamp" not in d.columns or "Demand_kW" not in d.columns:
        tc=[c for c in d if "time" in c.lower() or "date" in c.lower()]; dc=[c for c in d if "demand" in c.lower() or "load" in c.lower() or "kw" in c.lower()]
        if not tc or not dc: raise ValueError("A timestamp/date/time column and a demand/load kW column are required.")
        d=d[[tc[0],dc[0]]].rename(columns={tc[0]:"Timestamp",dc[0]:"Demand_kW"})
    d["Timestamp"]=pd.to_datetime(d["Timestamp"],errors="coerce"); d["Demand_kW"]=pd.to_numeric(d["Demand_kW"],errors="coerce")
    d=d.dropna(subset=["Timestamp","Demand_kW"]); d["Demand_kW"]=d["Demand_kW"].clip(lower=0)
    d=d.groupby("Timestamp",as_index=False)["Demand_kW"].mean().sort_values("Timestamp").set_index("Timestamp").resample("15min").mean()
    d["Demand_kW"]=d["Demand_kW"].interpolate(method="time").ffill().bfill(); return d

def validate_upload(raw):
    if raw is None or raw.empty: raise ValueError("The uploaded file is empty.")
    d=standardise(raw); days=(d.index.max()-d.index.min()).total_seconds()/86400
    if days<28: raise ValueError(f"At least 28 days are required. This file covers about {days:.1f} days.")
    return d,{"raw_rows":len(raw),"standardised_rows":len(d),"coverage_days":days,"start":d.index.min(),"end":d.index.max()}

def default_threshold(d):
    dev=d.iloc[:max(1,int(len(d)*.85))]["Demand_kW"]; daily=dev.resample("1D").max().dropna(); return float(max(1.0,(daily.quantile(.75) if len(daily) else dev.quantile(.75))))

def metrics(y,p,cap):
    y=np.asarray(y,float); p=np.asarray(p,float); ab=y>=cap; pb=p>=cap; tp=int(np.sum(ab&pb)); fp=int(np.sum((~ab)&pb)); fn=int(np.sum(ab&(~pb))); tn=int(np.sum((~ab)&(~pb)))
    prec=tp/(tp+fp) if tp+fp else None; rec=tp/(tp+fn) if tp+fn else None; far=fp/(tp+fp) if tp+fp else None; spec=tn/(tn+fp) if tn+fp else None
    f1=2*prec*rec/(prec+rec) if prec is not None and rec is not None and prec+rec else None; bal=(rec+spec)/2 if rec is not None and spec is not None else None
    return {"mae":float(mean_absolute_error(y,p)),"rmse":float(np.sqrt(mean_squared_error(y,p))),"r2":float(r2_score(y,p)),"recall":rec,"far":far,"f1":f1,"balanced_accuracy":bal,"tp":tp,"fp":fp,"fn":fn,"tn":tn}

def pct(v): return "N/A" if v is None else f"{100*v:.1f}%"

def split_times(d,is_upload):
    if is_upload:
        t=d.index[LOOKBACK:]; a=int(len(t)*.70); b=int(len(t)*.85); return t[:a],t[a:b],t[b:],"70/15/15 chronological split"
    return d.index[(d.index>=DATA_START)&(d.index<TRAIN_END)],d.index[(d.index>=TRAIN_END)&(d.index<VALID_END)],d.index[(d.index>=VALID_END)&(d.index<TEST_END)],"Fixed Optimise Prime EPA periods"

def sequences(scaled,index,times):
    pos=pd.Series(np.arange(len(index)),index=index); X=[]; ts=[]
    for t in times:
        if t in pos.index and int(pos[t])>=LOOKBACK:
            i=int(pos[t]); X.append(scaled[i-LOOKBACK:i]); ts.append(t)
    X=np.asarray(X,np.float32).reshape(-1,LOOKBACK,1); return X,pd.DatetimeIndex(ts)

def new_model():
    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(SEED)
    m=keras.Sequential([keras.layers.Input((LOOKBACK,1)),keras.layers.LSTM(48,activation="tanh",recurrent_activation="sigmoid",kernel_initializer=keras.initializers.GlorotUniform(seed=SEED),recurrent_initializer=keras.initializers.Orthogonal(seed=SEED)),keras.layers.Dense(24,activation="relu",kernel_initializer=keras.initializers.HeUniform(seed=SEED)),keras.layers.Dense(1)])
    m.compile(optimizer=keras.optimizers.Adam(.001),loss="mse"); return m
def recursive_forecast(model,xsc,ysc,history,start,hours):
    h=list(np.asarray(history,float)); out=[]; times=pd.date_range(start=start,periods=int(hours*4),freq="15min")
    for _ in times:
        w=np.asarray(h[-LOOKBACK:],float).reshape(-1,1); ws=xsc.transform(w).reshape(1,LOOKBACK,1); ps=model.predict(ws,verbose=0).reshape(-1,1); val=max(0.0,float(ysc.inverse_transform(ps)[0,0])); out.append(val); h.append(val)
    return times,np.asarray(out,float)

def train_lstm(name,cap,hours,raw=None,callback=None):
    def rep(p,m):
        if callback: callback(p,m)
    rep(5,"Loading and standardising data"); d=standardise(raw if raw is not None else load_real_depot_data(name)); is_up=raw is not None
    tr,va,te,strategy=split_times(d,is_up); rep(18,"Creating chronological train, validation and held-out test periods")
    dev=d.loc[d.index<=va.max(),["Demand_kW"]]; xsc=StandardScaler().fit(dev); ysc=StandardScaler().fit(dev); scaled=xsc.transform(d[["Demand_kW"]]).ravel()
    Xtr,ttr=sequences(scaled,d.index,tr); Xva,tva=sequences(scaled,d.index,va); Xte,tte=sequences(scaled,d.index,te)
    ytr=ysc.transform(d.loc[ttr,["Demand_kW"]]).ravel().astype(np.float32); yva=ysc.transform(d.loc[tva,["Demand_kW"]]).ravel().astype(np.float32); yte=d.loc[tte,"Demand_kW"].astype(float)
    rep(30,"Building selected LSTM model"); m=new_model(); stop=keras.callbacks.EarlyStopping(monitor="val_loss",patience=5,min_delta=1e-5,restore_best_weights=True)
    rep(38,"Training LSTM with early stopping"); t0=time.perf_counter(); hist=m.fit(Xtr,ytr,validation_data=(Xva,yva),epochs=EPOCHS,batch_size=BATCH,shuffle=False,verbose=0,callbacks=[stop]); seconds=time.perf_counter()-t0
    rep(78,"Generating held-out predictions"); ps=m.predict(Xte,batch_size=BATCH,verbose=0).reshape(-1,1); preds=np.clip(ysc.inverse_transform(ps).ravel(),0,None)
    base=np.asarray([float(d.loc[t-pd.Timedelta(days=1),"Demand_kW"]) if t-pd.Timedelta(days=1) in d.index else np.nan for t in tte]); mask=np.isfinite(base); tte=tte[mask]; yte=yte.iloc[np.where(mask)[0]]; preds=preds[mask]; base=base[mask]
    ml=metrics(yte.values,preds,cap); mb=metrics(yte.values,base,cap); rep(90,f"Generating {hours}-hour recursive LSTM forecast"); ft,ff=recursive_forecast(m,xsc,ysc,d["Demand_kW"].values,d.index.max()+pd.Timedelta(minutes=15),hours)
    meta={"model_type":"LSTM Depot Demand Forecast Model","selected_model":"LSTM","selected_depot":name,"dataset_source":"Uploaded depot telemetry" if is_up else "UK Power Networks Optimise Prime WS2 depot trial","split_strategy":strategy,"sampling_interval":"15 minutes","lookback":"24 hours / 96 intervals","train_period":f"{ttr.min():%Y-%m-%d} to {ttr.max():%Y-%m-%d}","validation_period":f"{tva.min():%Y-%m-%d} to {tva.max():%Y-%m-%d}","test_period":f"{tte.min():%Y-%m-%d} to {tte.max():%Y-%m-%d}","test_observations":int(len(tte)),"training_seconds":seconds,"metrics_lstm":ml,"metrics_baseline":mb,"historical_dataset_note":"Forecast status is based on historical project data and does not represent live grid telemetry."}
    rep(100,"Complete"); return {"metadata":meta,"history":d["Demand_kW"],"y_test":pd.Series(yte.values,index=tte),"lstm_preds":preds,"baseline_preds":base,"future_times":ft,"future_forecast":ff,"model":m,"x_scaler":xsc,"y_scaler":ysc,"training_history":hist.history}

@st.cache_data(show_spinner=False)
def load_final_artifacts():
    if not (METRICS_FILE.exists() and PREDS_FILE.exists() and FUTURE_FILE.exists()): return None
    return {"metrics":pd.read_csv(METRICS_FILE),"preds":pd.read_csv(PREDS_FILE,parse_dates=["timestamp"]),"future":pd.read_csv(FUTURE_FILE,parse_dates=["timestamp"]),"selection":pd.read_csv(SELECT_FILE) if SELECT_FILE.exists() else pd.DataFrame()}

def precomputed_bexleyheath(hours):
    a=load_final_artifacts()
    if a is None: return None
    d=standardise(load_real_depot_data("Bexleyheath")); m=a["metrics"]; p=a["preds"]; f=a["future"].head(int(hours*4)); lr=m[m.Model=="LSTM"].iloc[0]; br=m[m.Model=="Previous-day same-time"].iloc[0]
    def rm(r): return {"mae":float(r.MAE_kW),"rmse":float(r.RMSE_kW),"r2":float(r.R2),"recall":float(r.Breach_Recall),"far":float(r.False_Alarm_Ratio),"f1":float(r.F1_Score),"balanced_accuracy":float(r.Balanced_Accuracy),"tp":int(r.TP),"fp":int(r.FP),"fn":int(r.FN),"tn":int(r.TN)}
    return {"metadata":{"model_type":"LSTM Depot Demand Forecast Model","selected_model":"LSTM","selected_depot":"Bexleyheath","dataset_source":"UK Power Networks Optimise Prime WS2 depot trial","split_strategy":"Fixed Optimise Prime EPA periods","sampling_interval":"15 minutes","lookback":"24 hours / 96 intervals","train_period":"08 Jul 2021 to 31 Mar 2022","validation_period":"01 Apr 2022 to 09 May 2022","test_period":"10 May 2022 to 25 Jul 2022","test_observations":len(p),"training_seconds":float(lr.Training_Time_Seconds),"metrics_lstm":rm(lr),"metrics_baseline":rm(br),"historical_dataset_note":"Forecast status is based on the historical project dataset and does not represent live grid telemetry.","precomputed_final_epa_artifacts":True},"history":d["Demand_kW"],"y_test":pd.Series(p.actual_kw.values,index=p.timestamp),"lstm_preds":p.lstm_kw.values,"baseline_preds":p.baseline_kw.values,"future_times":pd.DatetimeIndex(f.timestamp),"future_forecast":f.lstm_forecast_kw.values,"model":None,"x_scaler":None,"y_scaler":None,"training_history":None,"comparison_metrics":m,"selection_decision":a["selection"]}

@st.cache_resource(show_spinner=False)
def builtin_pipeline(name,cap,hours):
    if name=="Bexleyheath":
        p=precomputed_bexleyheath(hours)
        if p is not None: return p
    return train_lstm(name,cap,hours)

def registry_load():
    if not REG_FILE.exists(): return {}
    try:
        r=json.loads(REG_FILE.read_text()); return {k:v for k,v in r.items() if isinstance(v,dict) and Path(v.get("data_file","x")).exists()}
    except Exception: return {}

def registry_write(r):
    STATE.mkdir(exist_ok=True); UP_DIR.mkdir(parents=True,exist_ok=True); REG_FILE.write_text(json.dumps(r,indent=2))

def save_upload(name,cap,raw,filename):
    if name in REAL_DEPOTS: raise ValueError("Choose a different name; built-in depot names are protected.")
    r=registry_load(); UP_DIR.mkdir(parents=True,exist_ok=True); ident=hashlib.sha256(name.lower().encode()).hexdigest()[:16]; p=UP_DIR/f"{ident}.csv"; raw.to_csv(p,index=False); r[name]={"capacity_kw":float(cap),"warning_kw":float(cap)*WARNING_ALPHA,"filename":filename,"data_file":str(p)}; registry_write(r); return r

def remove_upload(name):
    r=registry_load(); meta=r.pop(name,None)
    if meta:
        try: Path(meta["data_file"]).unlink(missing_ok=True)
        except Exception: pass
    registry_write(r); return r

def load_upload(name,r):
    return pd.read_csv(r[name]["data_file"]) if name in r else None

def evaluate_unseen(model,xsc,ysc,raw,cap):
    d=standardise(raw); scaled=xsc.transform(d[["Demand_kW"]]).ravel(); X,t=sequences(scaled,d.index,d.index[LOOKBACK:]); ps=model.predict(X,batch_size=BATCH,verbose=0).reshape(-1,1); pred=np.clip(ysc.inverse_transform(ps).ravel(),0,None); y=d.loc[t,"Demand_kW"].values; base=np.asarray([float(d.loc[x-pd.Timedelta(days=1),"Demand_kW"]) if x-pd.Timedelta(days=1) in d.index else np.nan for x in t]); mask=np.isfinite(base); return {"timestamps":t[mask],"actual":y[mask],"predictions":pred[mask],"baseline":base[mask],"lstm_metrics":metrics(y[mask],pred[mask],cap),"baseline_metrics":metrics(y[mask],base[mask],cap)}

persisted=registry_load()
defs={"authenticated":False,"current_depot":"Bexleyheath","uploaded_depots_registry":persisted,"grid_limit":17.0,"warn_threshold":13.6,"forecast_horizon":24,"unseen_eval":None,"retrained_pipeline":None,"archive":[]}
for k,v in defs.items():
    if k not in st.session_state: st.session_state[k]=v

if not st.session_state.authenticated:
    _,c,_=st.columns([1,2,1])
    with c:
        st.markdown("<div style='height:80px'></div><div style='text-align:center'><h2>⚡ Optivolt EV Depot Intelligence Portal</h2><p>Predictive Demand Forecasting & Capacity Management</p></div>",unsafe_allow_html=True)
        with st.container(border=True):
            st.text_input("Username or corporate email",value="operations@optivolt.co.uk"); st.text_input("Password",type="password",value="••••••••")
            if st.button("Authenticate Workspace Access",use_container_width=True): st.session_state.authenticated=True; st.rerun()
    st.stop()

with st.sidebar:
    st.markdown(f"### {ORG_NAME}\nData: UK Power Networks Optimise Prime  \nSelected model: **LSTM**")
    reg=st.session_state.uploaded_depots_registry; opts=REAL_DEPOTS+sorted([x for x in reg if x not in REAL_DEPOTS]); cur=st.session_state.current_depot if st.session_state.current_depot in opts else "Bexleyheath"; sel=st.selectbox("Select Active Depot Node:",opts,index=opts.index(cur))
    if sel!=st.session_state.current_depot: st.session_state.current_depot=sel; st.session_state.unseen_eval=None; st.session_state.retrained_pipeline=None; st.rerun()
    if sel in DEPOT_CAPACITY_KW: st.session_state.grid_limit=float(DEPOT_CAPACITY_KW[sel]); st.session_state.warn_threshold=WARNING_ALPHA*st.session_state.grid_limit
    else: st.session_state.grid_limit=float(reg[sel]["capacity_kw"]); st.session_state.warn_threshold=float(reg[sel].get("warning_kw",WARNING_ALPHA*st.session_state.grid_limit))
    st.markdown("---"); nav=st.radio("Navigation Menu:",["🏠 Home / Overview","🏢 Depots Setup","📊 Upload Depot Data","📈 Demand Forecast","🚗 Scenario Analysis","⚙️ Model Performance","🚨 System Alerts","📋 Reports","📜 Archive History","⚙️ Settings / Admin"])
def get_active():
    name=st.session_state.current_depot; cap=float(st.session_state.grid_limit); hours=int(st.session_state.forecast_horizon)
    if name in st.session_state.uploaded_depots_registry:
        if st.session_state.retrained_pipeline is not None and st.session_state.retrained_pipeline["metadata"]["selected_depot"]==name: return st.session_state.retrained_pipeline
        d=standardise(load_upload(name,st.session_state.uploaded_depots_registry)); return {"metadata":{"model_type":"LSTM not yet retrained for this uploaded depot","selected_model":"LSTM","selected_depot":name,"dataset_source":"Uploaded depot telemetry","historical_dataset_note":"Forecast status is based on uploaded historical data and does not represent live grid telemetry."},"history":d.Demand_kW,"y_test":pd.Series(dtype=float),"lstm_preds":np.array([]),"baseline_preds":np.array([]),"future_times":pd.DatetimeIndex([]),"future_forecast":np.array([])}
    return builtin_pipeline(name,cap,hours)

try:
    with st.spinner(f"Preparing {st.session_state.current_depot} forecasting evidence..."): pipe=get_active()
except Exception as e:
    st.error(str(e)); st.info("Check that the required Optimise Prime CSV files are present in the repository."); st.stop()

meta=pipe["metadata"]; hist=pipe["history"]; ytest=pipe["y_test"]; lp=np.asarray(pipe["lstm_preds"]); bp=np.asarray(pipe["baseline_preds"]); ft=pipe["future_times"]; ff=np.asarray(pipe["future_forecast"])
latest=float(hist.iloc[-1]); peak=float(np.max(ff)) if len(ff) else np.nan; cap=float(st.session_state.grid_limit); warn=float(st.session_state.warn_threshold)
if len(ff):
    breaches=np.where(ff>=cap)[0]; warnings=np.where(ff>=warn)[0]
    if len(breaches): status="CRITICAL RISK"; badge="danger"; breach_time=f"+{(breaches[0]+1)*.25:.2f} hours"; first=warnings[0] if len(warnings) else breaches[0]; lead=f"{max(0,(breaches[0]-first)*.25):.2f} hours"
    elif len(warnings): status="WARNING THRESHOLD FORECAST"; badge="warn"; breach_time="No capacity breach predicted"; lead="N/A"
    else: status="OPTIMAL SAFE"; badge="safe"; breach_time="No capacity breach predicted"; lead="N/A"
else: status="MODEL NOT RETRAINED"; badge="warn"; breach_time="No forecast available"; lead="N/A"

if nav=="🏠 Home / Overview":
    st.title(f"🏢 {st.session_state.current_depot} Operations Dashboard")
    st.markdown(f"<div class='perf'><b>Organisation:</b> {ORG_NAME}&nbsp;&nbsp;&nbsp; <b>Status:</b> <span class='badge {badge}'>{status}</span>&nbsp;&nbsp;&nbsp; <b>Expected Capacity Breach:</b> {breach_time}</div>",unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4); c1.metric("Latest Observed Demand",f"{latest:.1f} kW"); c2.metric("Predicted 24h Peak",f"{peak:.1f} kW" if np.isfinite(peak) else "N/A"); c3.metric("Planning Capacity Threshold",f"{cap:.1f} kW"); c4.metric("Warning Lead Time",lead)
    st.caption(meta.get("historical_dataset_note","Forecast status is based on the historical project dataset and does not represent live grid telemetry."))
    recent=hist.iloc[-7*STEPS_PER_DAY:]; fig=go.Figure(); fig.add_trace(go.Scatter(x=recent.index,y=recent.values,name="Observed Demand")); fig.add_hline(y=warn,line_dash="dot",annotation_text="Warning Threshold"); fig.add_hline(y=cap,line_dash="dash",annotation_text="Planning Capacity Threshold"); fig.update_layout(template="plotly_white",height=380,yaxis_title="Demand (kW)"); st.plotly_chart(fig,use_container_width=True)

elif nav=="🏢 Depots Setup":
    st.title("🏢 Depot Configuration Registry"); rows=[]
    for d in REAL_DEPOTS: rows.append({"Depot":d,"Type":"Built-in Optimise Prime","Planning Capacity Threshold (kW)":DEPOT_CAPACITY_KW[d],"Warning Threshold (kW)":WARNING_ALPHA*DEPOT_CAPACITY_KW[d]})
    for n,m in st.session_state.uploaded_depots_registry.items(): rows.append({"Depot":n,"Type":"Uploaded","Planning Capacity Threshold (kW)":m["capacity_kw"],"Warning Threshold (kW)":m.get("warning_kw",WARNING_ALPHA*m["capacity_kw"])})
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    if st.session_state.uploaded_depots_registry:
        rem=st.selectbox("Uploaded depot to remove",sorted(st.session_state.uploaded_depots_registry))
        if st.button("Remove Selected Uploaded Depot"):
            st.session_state.uploaded_depots_registry=remove_upload(rem)
            if st.session_state.current_depot==rem: st.session_state.current_depot="Bexleyheath"
            st.rerun()

elif nav=="📊 Upload Depot Data":
    st.title("📊 Upload Depot Data"); st.caption("Evaluate the existing LSTM on unseen uploaded data first; retrain only if adaptation is needed.")
    u=st.file_uploader("Upload CSV or Excel telemetry",type=["csv","xlsx"])
    if u is not None:
        try:
            raw=pd.read_csv(u) if u.name.lower().endswith(".csv") else pd.read_excel(u); std,rep=validate_upload(raw); st.success("Upload validation passed.")
            a,b,c=st.columns(3); a.metric("Raw rows",f"{rep['raw_rows']:,}"); b.metric("Standardised rows",f"{rep['standardised_rows']:,}"); c.metric("Coverage",f"{rep['coverage_days']:.1f} days")
            name=st.text_input("New depot name",value=Path(u.name).stem.replace("_"," ").title()); dcap=st.number_input("Planning Capacity Threshold (kW)",min_value=1.0,value=float(round(default_threshold(std),1)),step=1.0)
            st.markdown("#### Step 1 — Evaluate existing LSTM on unseen data")
            if st.button("Evaluate Existing LSTM on Unseen Data"):
                pbar=st.progress(0); stat=st.empty(); cb=lambda p,m:(pbar.progress(int(p)),stat.caption(m))
                with st.spinner("Preparing the existing project LSTM..."):
                    base=train_lstm("Bexleyheath",DEPOT_CAPACITY_KW["Bexleyheath"],24,callback=cb); ev=evaluate_unseen(base["model"],base["x_scaler"],base["y_scaler"],raw,float(dcap)); st.session_state.unseen_eval=ev
                pbar.progress(100)
            ev=st.session_state.unseen_eval
            if ev is not None:
                lm,bm=ev["lstm_metrics"],ev["baseline_metrics"]; table=pd.DataFrame({"Metric":["MAE (kW)","RMSE (kW)","R²","Breach Recall","False Alarm Ratio"],"Existing LSTM":[f"{lm['mae']:.3f}",f"{lm['rmse']:.3f}",f"{lm['r2']:.3f}",pct(lm['recall']),pct(lm['far'])],"Previous-Day Baseline":[f"{bm['mae']:.3f}",f"{bm['rmse']:.3f}",f"{bm['r2']:.3f}",pct(bm['recall']),pct(bm['far'])]}); st.table(table.set_index("Metric"))
                fig=go.Figure(); fig.add_trace(go.Scatter(x=ev["timestamps"],y=ev["actual"],name="Actual")); fig.add_trace(go.Scatter(x=ev["timestamps"],y=ev["predictions"],name="Existing LSTM")); fig.add_trace(go.Scatter(x=ev["timestamps"],y=ev["baseline"],name="Previous-Day Baseline",line=dict(dash="dash"))); fig.update_layout(template="plotly_white",height=390); st.plotly_chart(fig,use_container_width=True)
                st.markdown("#### Step 2 — Retrain only if adaptation is needed")
                if st.button("Retrain LSTM Using Uploaded Data"):
                    pbar=st.progress(0); stat=st.empty(); cb=lambda p,m:(pbar.progress(int(p)),stat.caption(m))
                    with st.spinner("Retraining LSTM on the uploaded depot..."):
                        r=train_lstm(name,float(dcap),int(st.session_state.forecast_horizon),raw=raw,callback=cb); st.session_state.retrained_pipeline=r; st.session_state.uploaded_depots_registry=save_upload(name,float(dcap),raw,u.name)
                    st.success(f"{name} saved. Select it from the sidebar.")
        except Exception as e: st.error(f"Upload validation failed: {e}")

elif nav=="📈 Demand Forecast":
    st.title("📈 Historical Demand & 24-Hour LSTM Forecast")
    if not len(ff): st.warning("This uploaded depot has not yet been retrained.")
    else:
        st.caption("The future forecast is recursive: each 15-minute LSTM prediction becomes input to the next step. No future actual demand is used.")
        h=hist.iloc[-192:]; fig=go.Figure(); fig.add_trace(go.Scatter(x=h.index,y=h.values,name="Observed Demand")); fig.add_trace(go.Scatter(x=ft,y=ff,name="LSTM Recursive Forecast",line=dict(dash="dash"))); fig.add_hline(y=warn,line_dash="dot",annotation_text="Warning Threshold"); fig.add_hline(y=cap,line_dash="dash",annotation_text="Planning Capacity Threshold"); fig.add_vline(x=ft[0],line_dash="dot",annotation_text="Forecast Start"); fig.update_layout(template="plotly_white",height=460,yaxis_title="Demand (kW)",legend=dict(orientation="h",y=1.12)); st.plotly_chart(fig,use_container_width=True); st.caption("Forecast status is based on the historical project dataset and does not represent live grid telemetry.")

elif nav=="🚗 Scenario Analysis":
    st.title("🚗 Scenario Analysis")
    if not len(ff): st.warning("A retrained LSTM forecast is required first.")
    else:
        st.caption("Scenario values are synthetic adjustments applied to the genuine LSTM forecast; they are not observed data."); growth=st.slider("Projected Fleet / Demand Growth (%)",0,100,20,5); sy=ff*(1+growth/100); sp=float(np.max(sy)); c1,c2,c3=st.columns(3); c1.metric("Scenario Peak",f"{sp:.1f} kW"); c2.metric("Warning Threshold",f"{warn:.1f} kW"); c3.metric("Planning Capacity Threshold",f"{cap:.1f} kW")
        fig=go.Figure(); fig.add_trace(go.Scatter(x=ft,y=ff,name="Base LSTM Forecast",line=dict(dash="dot"))); fig.add_trace(go.Scatter(x=ft,y=sy,name="Scenario-Adjusted Forecast")); fig.add_hline(y=warn,line_dash="dash"); fig.add_hline(y=cap,line_dash="dash"); fig.update_layout(template="plotly_white",height=430,yaxis_title="Demand (kW)"); st.plotly_chart(fig,use_container_width=True)
        if sp>=cap: st.warning("The hypothetical scenario exceeds the Planning Capacity Threshold.")
        elif sp>=warn: st.info("The hypothetical scenario reaches the warning threshold but remains below the Planning Capacity Threshold.")
        else: st.success("The hypothetical scenario remains below the warning threshold.")

elif nav=="⚙️ Model Performance":
    st.title("⚙️ Model Performance"); comp=pipe.get("comparison_metrics"); seldec=pipe.get("selection_decision")
    if comp is not None:
        st.subheader("Final EPA Model Comparison — Bexleyheath"); st.caption("All approaches use the same 7,392 untouched held-out test timestamps. Baseline = previous-day same-time.")
        d=comp.copy(); d["MAE (kW)"]=d.MAE_kW.map(lambda x:f"{x:.3f}"); d["RMSE (kW)"]=d.RMSE_kW.map(lambda x:f"{x:.3f}"); d["R²"]=d.R2.map(lambda x:f"{x:.3f}"); d["Recall"]=d.Breach_Recall.map(lambda x:f"{100*x:.1f}%"); d["FAR"]=d.False_Alarm_Ratio.map(lambda x:f"{100*x:.1f}%"); d["F1"]=d.F1_Score.map(lambda x:f"{100*x:.1f}%"); d["Balanced Accuracy"]=d.Balanced_Accuracy.map(lambda x:f"{100*x:.1f}%"); d["Training Time"]=d.Training_Time_Seconds.map(lambda x:f"{x:.1f}s")
        st.dataframe(d[["Model","Role","MAE (kW)","RMSE (kW)","R²","Recall","FAR","F1","Balanced Accuracy","Training Time"]],use_container_width=True,hide_index=True)
        if seldec is not None and len(seldec):
            wins=seldec.Winner.value_counts(); x,y,z=st.columns(3); x.metric("LSTM Wins",int(wins.get("LSTM",0))); y.metric("Random Forest Wins",int(wins.get("Random Forest",0))); z.metric("XGBoost Wins",int(wins.get("XGBoost",0))); st.success("LSTM was selected as the final model: 6 of 8 criteria won. Random Forest won lowest false alarm ratio; XGBoost won fastest training.")
    if len(ytest) and len(lp):
        m=meta["metrics_lstm"]; st.subheader("Selected LSTM — Held-Out Test"); q1,q2,q3,q4=st.columns(4); q1.metric("MAE",f"{m['mae']:.3f} kW"); q2.metric("RMSE",f"{m['rmse']:.3f} kW"); q3.metric("R²",f"{m['r2']:.3f}"); q4.metric("Breach Recall",pct(m['recall']))
        fig=go.Figure(); fig.add_trace(go.Scatter(x=ytest.index,y=ytest.values,name="Actual Demand")); fig.add_trace(go.Scatter(x=ytest.index,y=lp,name="LSTM Prediction")); fig.update_layout(template="plotly_white",height=420,title="Actual vs LSTM Predicted Demand — Full Held-Out Test"); st.plotly_chart(fig,use_container_width=True)
        sc=go.Figure(); sc.add_trace(go.Scatter(x=ytest.values,y=lp,mode="markers",marker=dict(size=5,opacity=.35),name="Held-out observations")); mv=float(max(ytest.max(),lp.max())); sc.add_trace(go.Scatter(x=[0,mv],y=[0,mv],mode="lines",line=dict(dash="dash"),name="Ideal: Predicted = Actual")); sc.update_layout(template="plotly_white",height=500,xaxis_title="Actual Demand (kW)",yaxis_title="Predicted Demand (kW)"); st.plotly_chart(sc,use_container_width=True)
        residual=ytest.values-lp; hh=go.Figure(go.Histogram(x=residual,nbinsx=55)); hh.add_vline(x=0,line_dash="dash"); hh.add_vline(x=float(np.mean(residual)),line_dash="dot"); hh.add_vline(x=float(np.median(residual)),line_dash="dashdot"); hh.update_layout(template="plotly_white",height=400,title="LSTM Held-Out Residual Distribution",xaxis_title="Residual (Actual − Predicted) kW"); st.plotly_chart(hh,use_container_width=True)

elif nav=="🚨 System Alerts":
    st.title("🚨 System Alerts")
    if not len(ff): st.warning("No forecast is available for this uploaded depot yet.")
    elif np.any(ff>=cap): i=int(np.where(ff>=cap)[0][0]); st.error(f"Planning Capacity Threshold is forecast to be reached at {ft[i]:%d %b %Y %H:%M}.")
    elif np.any(ff>=warn): i=int(np.where(ff>=warn)[0][0]); st.warning(f"Warning Threshold is forecast to be reached at {ft[i]:%d %b %Y %H:%M}.")
    else: st.success("No warning or planning-capacity breach is forecast in the selected horizon.")
    st.caption("Forecast status is based on the historical project dataset and does not represent live grid telemetry.")

elif nav=="📋 Reports":
    st.title("📋 Reports"); st.write("Final comparison: previous-day same-time baseline, Random Forest, XGBoost and LSTM. Selected final model: LSTM.")
    for p,label in [(METRICS_FILE,"Model Comparison Metrics"),(SELECT_FILE,"Model Selection Decision"),(PREDS_FILE,"Held-Out Predictions"),(FUTURE_FILE,"LSTM 24-Hour Forecast")]:
        if p.exists(): st.download_button(f"Download {label} CSV",p.read_bytes(),file_name=p.name,mime="text/csv")
    st.subheader("Current depot metadata"); st.json(meta)

elif nav=="📜 Archive History":
    st.title("📜 Archive History")
    if st.session_state.archive: st.dataframe(pd.DataFrame(st.session_state.archive),use_container_width=True)
    else: st.info("No archived snapshots in this session.")
    if st.button("Archive Current Snapshot"):
        st.session_state.archive.append({"created_at":datetime.now().isoformat(timespec="seconds"),"depot":st.session_state.current_depot,"latest_observed_kw":latest,"predicted_peak_kw":peak if np.isfinite(peak) else None,"planning_capacity_threshold_kw":cap,"warning_threshold_kw":warn,"status":status,"selected_model":"LSTM"}); st.success("Snapshot archived.")

elif nav=="⚙️ Settings / Admin":
    st.title("⚙️ Settings / Admin"); st.text_input("Organisation",value=ORG_NAME,disabled=True); st.text_input("Selected operational model",value="LSTM",disabled=True); st.markdown(f"**Planning Capacity Threshold:** {cap:.1f} kW  \n**Warning Threshold:** {warn:.1f} kW  \n**Methodology:** LSTM selected; Random Forest and XGBoost retained as comparison models; previous-day same-time retained as operational baseline.  \n**Data provenance:** UK Power Networks Optimise Prime WS2 depot trial. Historical project data, not live grid telemetry.")
    if st.button("Sign Out"): st.session_state.authenticated=False; st.rerun()
