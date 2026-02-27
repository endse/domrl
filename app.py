import streamlit as st
import torch
import numpy as np
import pandas as pd
import time
import gymnasium as gym
import plotly.express as px
import plotly.graph_objects as go
import os

from domrl.env.rec_env import RealTimeRecEnv
from domrl.agent.sac import SACAgent
from domrl.agent.weight_agent import WeightAgent

import requests

# ==========================================
# 1. Configuration & Layout
# ==========================================
st.set_page_config(
    page_title="DOM-RL Command Center",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Enterprise" feel
st.markdown("""
<style>
    .stMetric {
        background-color: #0e1117;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #262730;
    }
    .stButton button {
        width: 100%;
        border-radius: 5px;
        height: 3em; 
    }
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1.1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. Core Logic & Caching
# ==========================================
CATEGORY_NAMES = [
    "Action", "Comedy", "Drama", "Sci-Fi", "Horror", 
    "Romance", "Documentary", "Thriller", "Fantasy", "Kids"
]

@st.cache_resource
def load_system():
    """Initializes the Environment and Agents (cached)."""
    env = RealTimeRecEnv(slate_size=3)
    
    # --- Movie Database ---
    from domrl.utils.movie_db import get_movie_db
    from domrl.config import cfg
    movie_db = get_movie_db(cfg.MOVIE_LENS_PATH)
    
    # --- SAC Agent ---
    sac_agent = SACAgent(
        state_dim=0, 
        action_dim=env.action_space.n,
        num_items=env.num_categories,
        hidden_dim=512
    )
    # Load Weights for SAC
    # Search for latest checkpoint or final
    import glob
    import os
    
    checkpoints = glob.glob(os.path.join(cfg.MODEL_CHECKPOINT_DIR, "actor_*.pth"))
    sac_path = None
    if checkpoints:
        # Sort by modification time (newest first) or by name if needed. 
        # By time is safer for "latest run".
        sac_path = max(checkpoints, key=os.path.getmtime)
        
    if sac_path and os.path.exists(sac_path):
        try:
             state_dict = torch.load(sac_path, map_location=torch.device('cpu'))
             sac_agent.actor.load_state_dict(state_dict)
             print(f"[System] Loaded SAC Model: {sac_path}")
        except Exception as e:
            print(f"[System] Warning: Could not load SAC model: {e}")
            
    # --- Weight Agent ---
    # Note: action_dim=10 for Embeddings
    weight_agent = WeightAgent(
        action_dim=10, 
        num_items=env.num_categories,
        hidden_dim=64
    )
    # Ideally load WeightAgent weights too if we saved them...
    # For now, it initializes random (which shows the UI structure at least)
    
    return env, sac_agent, weight_agent, sac_path, movie_db

@st.cache_data(show_spinner=False)
def fetch_real_poster(imdb_id_int, api_key):
    # Debugging logs to terminal
    if cfg.DEBUG:
        print(f"[DEBUG] Fetching poster. ID: {imdb_id_int}, KeyPresent: {bool(api_key)}")
    
    if not api_key or not imdb_id_int: 
        if cfg.DEBUG: print("[DEBUG] Missing Key or ID")
        return None
    try:
        # Format IMDb ID: tt + 7 digits (with leading zeros)
        imdb_str = f"tt{int(imdb_id_int):07d}"
        
        # Use Standard OMDb JSON API (more reliable than img.omdbapi.com)
        url = f"http://www.omdbapi.com/?i={imdb_str}&apikey={api_key}"
        
        if cfg.DEBUG: print(f"[DEBUG] Requesting Metadata: {url.replace(api_key, 'HIDDEN')}")
        resp = requests.get(url, timeout=2)
        
        if resp.status_code == 200:
            data = resp.json()
            if data.get('Response') == 'True':
                poster_url = data.get('Poster')
                if poster_url and poster_url != "N/A":
                    if cfg.DEBUG: print(f"[DEBUG] Found Poster: {poster_url}")
                    return poster_url
                else:
                     if cfg.DEBUG: print(f"[DEBUG] Poster field is N/A or missing")
            else:
                 if cfg.DEBUG: print(f"[DEBUG] OMDb Error: {data.get('Error')}")
        else:
            if cfg.DEBUG: print(f"[DEBUG] HTTP Error: {resp.status_code}")
            
    except Exception as e:
        if cfg.DEBUG: print(f"[DEBUG] Exception in fetch_real_poster: {e}")
        return None
    return None

def get_slate_data(slate_indices):
    from domrl.utils.movie_db import get_movie_db
    from domrl.config import cfg
    db = get_movie_db(cfg.MOVIE_LENS_PATH)
    
    data = []
    for idx_or_cat in slate_indices:
        # slate_indices are category indices (0-9)
        # Sample 1 movie (ID, Title) for this category
        items = db.sample_movies_with_id(idx_or_cat, n=1)
        mid, title = items[0]
        data.append({"id": mid, "title": title})
    return data

def get_slate_names(slate_indices):
    # Wrapper for legacy parts logging names
    data = get_slate_data(slate_indices)
    return [d['title'] for d in data]

# ==========================================
# 3. Session State Management
# ==========================================
if 'system_loaded' not in st.session_state:
    st.session_state.env, st.session_state.sac_agent, st.session_state.weight_agent, st.session_state.model_source, st.session_state.movie_db = load_system()
    st.session_state.system_loaded = True

if 'obs' not in st.session_state:
    st.session_state.obs, _ = st.session_state.env.reset()
    st.session_state.done = False
    st.session_state.step = 0
    st.session_state.total_reward = 0.0
    st.session_state.history = [] # For Analytics

# ==========================================
# 4. Sidebar: Mission Control
# ==========================================
with st.sidebar:
    st.title("🎛️ Mission Control")
    
    # Validation info
    if st.session_state.model_source:
        st.success(f"🧠 Model: {os.path.basename(st.session_state.model_source)}")
    else:
        st.warning("🧠 Model: Untrained/Random")
        
    if st.button("🔄 Reload Latest Model"):
        st.cache_resource.clear()
        # Clear session state to force reload
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
        
    st.markdown("---")
    
    # Persona Selection
    st.subheader("User Simulation")
    persona = st.selectbox(
        "Select Persona Profile", 
        ["Standard", "Binger", "Browser", "Critic"],
        help="Determines the hidden behavior dynamics of the simulated user."
    )
    persona_map = {"Standard": 0, "Binger": 1, "Browser": 2, "Critic": 3}
    
    # NEW: Context Controls
    st.subheader("User Context")
    c_ctx1, c_ctx2 = st.columns(2)
    mood_label = c_ctx1.selectbox("Mood", ["Neutral", "Happy", "Sad", "Tired"], help="Biases genre preferences.")
    mood_map = {"Neutral": 0, "Happy": 1, "Sad": 2, "Tired": 3}
    
    time_val = c_ctx2.slider("Hour (0-24)", 0, 24, 20, help="Time of Day affects genre preference (e.g. Horror at night).")
    
    # Update Environment Context
    if 'env' in st.session_state:
        st.session_state.env.set_user_context(mood_map[mood_label], float(time_val))
    
    # API Config
    st.subheader("External Services")
    # Using 'omdb_key' variable but storing in session state as generic 'api_key' or keeping 'tmdb_key' name for minimal refactor?
    # Better to rename for clarity.
    from domrl.config import cfg
    omdb_key = st.text_input("OMDb API Key", value=cfg.OMDB_API_KEY, type="password", help="Enter Key for Real Posters (img.omdbapi.com)")
    st.session_state.omdb_key = omdb_key
    
    # Debug Toggle
    debug_mode = st.toggle("🛠️ Debug Poster Fetching", value=False)
    st.session_state.debug_mode = debug_mode

    st.session_state.omdb_key = omdb_key
    
    if omdb_key:
        if st.button("Test Connection"):
            # Try fetching Toy Story (IMDb: tt0114709)
            try:
                # Use the function logic manually
                test_url = f"http://www.omdbapi.com/?i=tt0114709&apikey={omdb_key}"
                r = requests.get(test_url, timeout=2)
                if r.status_code == 200:
                    data = r.json()
                    if data.get('Response') == 'True':
                        poster = data.get('Poster')
                        st.success(f"✅ API Key is Valid! Found: {data.get('Title')}")
                        if poster and poster != 'N/A':
                             st.image(poster, width=150)
                        else:
                             st.warning("Key valid but movie has no poster.")
                    else:
                        st.error(f"❌ OMDb Error: {data.get('Error')}")
                else:
                    st.error(f"❌ HTTP Error: {r.status_code}")
            except Exception as e:
                st.error(f"❌ Connection Failed: {e}")

    # Manual Override Switch
    st.subheader("Objective Override")
    enable_override = st.toggle("Enable Manual Weights", value=False)
    
    overrides = []
    if enable_override:
        c1, c2 = st.columns(2)
        w_eng = c1.slider("Engagement", 0.0, 5.0, 1.0)
        w_sat = c2.slider("Satisfaction", 0.0, 5.0, 1.0)
        w_div = c1.slider("Diversity", 0.0, 5.0, 1.0)
        w_fair = c2.slider("Fairness", 0.0, 5.0, 1.0)
        overrides = np.array([w_eng, w_sat, w_div, w_fair], dtype=np.float32)
    
    st.markdown("---")
    # Simulation Utility
    if st.button("⏮️ Reset Episode", type="secondary"):
        st.session_state.obs, _ = st.session_state.env.reset(options={'persona_id': persona_map[persona]})
        st.session_state.done = False
        st.session_state.step = 0
        st.session_state.total_reward = 0.0
        st.session_state.history = []
        st.rerun()

    st.info(f"Step: {st.session_state.step} | Acc. Reward: {st.session_state.total_reward:.1f}")

# ==========================================
# 5. Main Dashboard Logic
# ==========================================
st.title("DOM-RL Enterprise Dashboard")

tab_live, tab_analysis, tab_internals = st.tabs(["🚀 Live Operations", "📈 Analytics Suite", "🧠 Model Internals"])

# Prepare Tensor Data
current_obs = st.session_state.obs
device = st.session_state.weight_agent.device
state_tensor = {}
for k, v in current_obs.items():
    if k in ['persona_id', 'history']:
        t = torch.as_tensor(v, device=device, dtype=torch.long)
        if t.dim() == 0: t = t.unsqueeze(0)
        if k == 'history' and t.dim() == 1: t = t.unsqueeze(0)
        state_tensor[k] = t
    else:
        state_tensor[k] = torch.FloatTensor(v).unsqueeze(0).to(device)

# --- Compute Weights ---
with torch.no_grad():
    suggested_weights_t = st.session_state.weight_agent.actor(state_tensor)
    suggested_weights = suggested_weights_t.cpu().numpy()[0]
    
effective_weights = overrides if enable_override else suggested_weights

# ------------------------------------------
# TAB 1: Live Operations
# ------------------------------------------
with tab_live:
    # Top Row: KPIs
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    sat_val = st.session_state.env.user_satisfaction
    enthusiasm = current_obs['user_features'][0]
    scroll = current_obs['micro_signals'][0]
    
    # Use real confidence if available
    if 'confidence' not in st.session_state: st.session_state.confidence = 0.0
    
    kpi1.metric("Satisfaction", f"{sat_val:.1%}", delta=f"{(sat_val-0.5)*100:.0f} pts")
    kpi2.metric("Enthusiasm", f"{enthusiasm:.2f}", help="Base probability of interaction")
    kpi3.metric("Scroll Velocity", f"{scroll:.1f}", delta="-High" if scroll > 8 else "Normal")
    kpi4.metric("Algorithm Confidence", f"{st.session_state.confidence:.1%}", help="Certainty of the policy network (Max Probability).")

    st.markdown("### 🎯 Current Objective Strategy")
    # Weight Visualization
    w_df = pd.DataFrame({
        "Objective": ["Engagement", "Satisfaction", "Diversity", "Fairness"],
        "Weight": effective_weights,
        "Source": ["Manual" if enable_override else "AI Agent"] * 4
    })
    
    fig_w = px.bar(
        w_df, x="Weight", y="Objective", orientation='h', 
        color="Objective", text_auto=True,
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    fig_w.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig_w, use_container_width=True)
    
    st.divider()
    
    # Recommendation Engine
    st.markdown("### 🎬 Recommendation Slate")
    
    if st.session_state.history:
        last_item = st.session_state.history[-1]
        
        cols = st.columns(3)
        
        # Try to use new rich object, fallback to names
        if 'slate_obj' in last_item:
            slate_data = last_item['slate_obj']
            # slate_data is list of {id, title}
            
            for i, (col, item) in enumerate(zip(cols, slate_data)):
                with col:
                    mid = item['id']
                    title = item['title']
                    
                    # Try Real Fetch
                    poster = None
                    debug_log = []
                    
                    # Look up IMDb ID
                    imdb_id = st.session_state.movie_db.imdb_map.get(mid)
                    debug_log.append(f"Movie ID: {mid} | IMDb ID: {imdb_id}")
                    
                    has_key = bool(st.session_state.get('omdb_key'))
                    debug_log.append(f"API Key Present: {has_key}")
                    
                    if has_key:
                         if imdb_id:
                            poster = fetch_real_poster(imdb_id, st.session_state.omdb_key)
                            status = "Success" if poster else "URL Gen Failed"
                            debug_log.append(f"Fetch Result: {status}")
                         else:
                            debug_log.append("Skipping: No IMDb ID mapped")
                    else:
                        debug_log.append("Using Fallback: No API Key")
                    
                    if not poster:
                         poster = f"https://placehold.co/400x600/101010/FFFFFF/png?text={title.replace(' ', '+')}"
                    
                    st.image(poster, use_container_width=True)
                    st.caption(f"**{title}**")
                    
                    # --- INTERACTION BUTTONS ---
                    b_col1, b_col2 = st.columns(2)
                    
                    # LIKE Button
                    if b_col1.button("👍", key=f"like_{mid}_{st.session_state.step}", help="I like this! (Boosts Enthusiasm)"):
                        # 1. Update Environment State Directly (Inject Reality)
                        # CRITICAL FIX: Map Movie ID -> Category Index for the Agent's Embedding Layer
                        cat_idx = st.session_state.movie_db.movie_cat_map.get(mid, 0) # Default to 0
                        
                        st.session_state.env.history = np.roll(st.session_state.env.history, -1)
                        st.session_state.env.history[-1] = cat_idx 
                        
                        # Boost Enthusiasm
                        current_enth = st.session_state.env.user_state[0]
                        st.session_state.env.user_state[0] = min(1.0, current_enth + 0.2)
                        
                        # 2. Log User Feedback
                        st.toast(f"You liked {title}! Recommendations updating...", icon="🎉")
                        if 'interactions' not in st.session_state: st.session_state.interactions = []
                        st.session_state.interactions.append({
                            "step": st.session_state.step,
                            "time": time.strftime("%H:%M:%S"),
                            "movie": title,
                            "action": "Like",
                            "reward": 5.0
                        })
                        
                        # 3. Accumulated Reward for this manual step
                        st.session_state.total_reward += 5.0
                        
                        # 4. Advance Step (Manually) and Update Obs
                        st.session_state.env.current_step += 1
                        st.session_state.step += 1
                        st.session_state.obs = st.session_state.env._get_obs()
                        
                        # 5. Rerun to generate NEW recommendations based on this Like
                        st.rerun()

                    # DISLIKE Button
                    if b_col2.button("👎", key=f"dislike_{mid}_{st.session_state.step}", help="Not for me. (Reduces Enthusiasm)"):
                        # Penalize Enthusiasm
                        current_enth = st.session_state.env.user_state[0]
                        st.session_state.env.user_state[0] = max(0.0, current_enth - 0.2)
                        
                        st.toast(f"You disliked {title}.", icon="📉")
                        if 'interactions' not in st.session_state: st.session_state.interactions = []
                        st.session_state.interactions.append({
                            "step": st.session_state.step,
                            "time": time.strftime("%H:%M:%S"),
                            "movie": title,
                            "action": "Dislike",
                            "reward": -1.0
                        })
                        
                        st.session_state.total_reward -= 1.0
                        st.session_state.env.current_step += 1
                        st.session_state.step += 1
                        st.session_state.obs = st.session_state.env._get_obs()
                        st.rerun()

                    if st.session_state.get('debug_mode', False):
                        with st.expander("Debug Info"):
                            for log in debug_log:
                                st.text(log)
        else:
            slate_names = eval(last_item['slate']) # Legacy
            for i, (col, item_name) in enumerate(zip(cols, slate_names)):
                with col:
                    poster_url = f"https://placehold.co/400x600/101010/FFFFFF/png?text={item_name.replace(' ', '+')}"
                    st.image(poster_url, use_container_width=True)
                    st.caption(f"**{item_name}**")
                    if st.session_state.get('debug_mode', False):
                        st.caption("Legacy Mode (No ID available)")
                
        # Only show this success message if it was a distinct step? 
        # Actually, if we just reran, this will show the result of the LAST interaction.
        last_interaction = None
        if 'interactions' in st.session_state and st.session_state.interactions:
            last_interaction = st.session_state.interactions[-1]
            if last_interaction['step'] == st.session_state.step - 1:
                 st.success(f"Feedback Recorded: **{last_interaction['action']}** on **{last_interaction['movie']}**")
        
        # Keep the old msg for auto-steps
        if not last_interaction or last_interaction['step'] != st.session_state.step - 1:
             st.success(f"User Action: Clicked **{last_item.get('chosen_item_name', 'None')}** (+{last_item['reward']:.2f} Reward)")
    else:
        st.warning("Waiting for initialization...")

    # Action Button
    pad1, btn_col, pad2 = st.columns([1, 2, 1])
    if not st.session_state.done:
        if btn_col.button("⚡ EXECUTE NEXT STEP", type="primary"):
            # Update Env Weights
            st.session_state.env.weights = effective_weights
            st.session_state.obs['weights'] = effective_weights
            state_tensor['weights'] = torch.FloatTensor(effective_weights).unsqueeze(0).to(device)
            
            # Sample Action
            with torch.no_grad():
                action_idx, probs, _ = st.session_state.sac_agent.actor.sample(state_tensor)
                action_idx = action_idx.item()
                
                # Update Confidence Metric
                st.session_state.confidence = probs.max().item()
            
            # Step Env
            # Map action to slate first for logging
            # HERE: We generate the slate data (IDs and Titles)
            slate_indices = st.session_state.env.slate_mapper.get_slate(action_idx)
            slate_data = get_slate_data(slate_indices) # List of dicts {id, title}
            slate_names = [d['title'] for d in slate_data]
            
            next_obs, reward_vec, terminated, truncated, _ = st.session_state.env.step(action_idx)
            
            # Scalarize for metric
            scalar_reward = np.dot(reward_vec, effective_weights)
            
            # Log
            chosen_idx = int(st.session_state.env.history[-1])
            st.session_state.total_reward += scalar_reward
            st.session_state.history.append({
                "step": st.session_state.step,
                "slate": str(slate_names), # Legacy: keep as string of names
                "slate_obj": slate_data,   # NEW: Store rich data
                "reward": float(scalar_reward), 
                "satisfaction": sat_val,
                "chosen_item_name": slate_names[0], 
                "w_eng": effective_weights[0],
                "w_sat": effective_weights[1]
            })
            
            st.session_state.obs = next_obs
            st.session_state.done = terminated or truncated
            st.session_state.step += 1
            st.rerun()
    else:
        st.error("Session Ended. Please Reset.")

# ------------------------------------------
# TAB 2: Analytics Suite
# ------------------------------------------
with tab_analysis:
    st.subheader("📊 Session Analytics")

    # NEW: Interaction Log
    if 'interactions' in st.session_state and st.session_state.interactions:
        st.info("Human Feedback Recorded")
        st.dataframe(pd.DataFrame(st.session_state.interactions).iloc[::-1], use_container_width=True) # Show newest first
        st.divider()
    
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        
        # 1. Satisfaction Trend
        fig_sat = px.line(df, x="step", y="satisfaction", title="User Satisfaction Over Time", markers=True)
        fig_sat.add_hline(y=0.0, line_dash="dash", line_color="red", annotation_text="Churn Threshold")
        st.plotly_chart(fig_sat, use_container_width=True)
        
        # 2. Weight Dynamics
        st.subheader("Strategy Adaptation")
        fig_dyn = go.Figure()
        fig_dyn.add_trace(go.Scatter(x=df['step'], y=df['w_eng'], mode='lines', name='Engagement Weight'))
        fig_dyn.add_trace(go.Scatter(x=df['step'], y=df['w_sat'], mode='lines', name='Satisfaction Weight'))
        st.plotly_chart(fig_dyn, use_container_width=True)
        
        # 3. Data Table
        with st.expander("Raw Session Data"):
            st.dataframe(df)
    else:
        st.info("Run the simulation to generate analytics data.")

# ------------------------------------------
# TAB 3: Model Internals
# ------------------------------------------
with tab_internals:
    st.subheader("🧠 Neural Diagnostics")
    
    c_in1, c_in2 = st.columns(2)
    
    with c_in1:
        st.markdown("#### Input State Vector")
        st.json({
            "User Features (Enthusiasm, etc)": current_obs['user_features'].tolist(),
            "Micro Signals (Scroll, etc)": current_obs['micro_signals'].tolist(),
            "History (Last 10 Items)": current_obs['history'].tolist()
        })
        
    with c_in2:
        st.markdown("#### Risk Assessment (CVaR)")
        st.caption("The Weight Agent predicts the distribution of future returns. High spread = High Uncertainty.")
        
        # Visualize Quantiles from Meta-Critic
        state_pred = state_tensor.copy()
        # Need to use the weights in the tensor:
        state_pred['weights'] = torch.FloatTensor(effective_weights).unsqueeze(0).to(device)
        
        with torch.no_grad():
            quantiles = st.session_state.weight_agent.critic(state_pred)
            quantiles_np = quantiles.cpu().numpy()[0]
        
        fig_dist = px.histogram(x=quantiles_np, nbins=20, title="Predicted Return Distribution")
        st.plotly_chart(fig_dist, use_container_width=True)
        
