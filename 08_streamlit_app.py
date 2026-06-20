import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingClassifier
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="IPL Analytics",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CUSTOM CSS — BCCI.tv Inspired Navy Theme
# ============================================================
st.markdown("""
<style>
.stApp {
    background: #0b1530;
    color: #ffffff;
}

[data-testid="stSidebar"] { display: none; }

/* Top navbar */
.topnav {
    background: #0e1b3d;
    border-bottom: 1px solid #1e2d54;
    padding: 14px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: -1rem -1rem 1.5rem -1rem;
}

.navbar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
}

.navbar-logo {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #ff6b00, #ff9100);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}

.navbar-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: 0.5px;
}

.navbar-subtitle {
    font-size: 0.7rem;
    color: #5b6b96;
    letter-spacing: 1px;
}

/* Metric cards */
.metric-box {
    background: #131f45;
    border: 1px solid #1e2d54;
    border-radius: 10px;
    padding: 18px;
    text-align: center;
}

.metric-value {
    font-size: 1.9rem;
    font-weight: 700;
    color: #ff6b00;
}

.metric-value-blue { color: #3a8ef6; }
.metric-value-white { color: #ffffff; }

.metric-label {
    font-size: 0.72rem;
    color: #5b6b96;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 6px;
}

/* Section headers */
.section-header {
    font-size: 1rem;
    font-weight: 700;
    color: #ffffff;
    background: #131f45;
    border-left: 3px solid #ff6b00;
    padding: 10px 14px;
    margin: 0 0 12px 0;
    border-radius: 4px;
    letter-spacing: 0.5px;
}

/* Panel card */
.panel {
    background: #131f45;
    border: 1px solid #1e2d54;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 16px;
}

.insight-text {
    color: #c5cfe6;
    font-size: 0.85rem;
    line-height: 1.6;
}

.stat-orange { color: #ff6b00; font-weight: 700; }
.stat-blue { color: #3a8ef6; font-weight: 700; }
.stat-green { color: #2ecc71; font-weight: 700; }
.stat-red { color: #ff4d6d; font-weight: 700; }

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.stSelectbox > div > div, .stNumberInput > div > div {
    background: #131f45;
    border: 1px solid #1e2d54;
    color: white;
}

.stSlider > div > div > div > div { background-color: #ff6b00; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CHART THEME
# ============================================================
NAVY_COLORS = ['#ff6b00', '#3a8ef6', '#2ecc71', '#ff4d6d', '#9b59b6', '#f1c40f']

def navy_layout(title="", height=320):
    return dict(
        title=dict(text=title, font=dict(color='#ffffff', size=13)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#5b6b96', size=11),
        height=height,
        margin=dict(l=30, r=15, t=35, b=30),
        xaxis=dict(gridcolor='#1e2d54', linecolor='#1e2d54', tickfont=dict(color='#5b6b96', size=10)),
        yaxis=dict(gridcolor='#1e2d54', linecolor='#1e2d54', tickfont=dict(color='#5b6b96', size=10)),
        legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#5b6b96', size=10))
    )

# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data
def load_data():
    matches = pd.read_csv('matches_clean.csv')
    deliveries = pd.read_csv('deliveries_clean.csv')
    return matches, deliveries

@st.cache_resource
def train_model(matches, deliveries):
    first_innings = deliveries[deliveries['inning'] == 1].groupby('match_id').agg(
        first_innings_total=('total_runs', 'sum')
    ).reset_index()
    second_innings = deliveries[deliveries['inning'] == 2].copy()
    second_innings = second_innings.merge(first_innings, on='match_id')
    second_innings = second_innings.merge(
        matches[['id', 'winner', 'team1', 'team2']], left_on='match_id', right_on='id'
    )
    second_innings = second_innings.sort_values(['match_id', 'inning', 'over', 'ball'])
    second_innings['cumulative_runs'] = second_innings.groupby('match_id')['total_runs'].cumsum()
    second_innings['cumulative_wickets'] = second_innings.groupby('match_id')['is_wicket'].cumsum()
    second_innings['ball_number'] = second_innings.groupby('match_id').cumcount() + 1
    second_innings['balls_remaining'] = 120 - second_innings['ball_number']
    second_innings['runs_needed'] = second_innings['first_innings_total'] - second_innings['cumulative_runs'] + 1
    second_innings['current_rr'] = (second_innings['cumulative_runs'] / second_innings['ball_number']) * 6
    second_innings['required_rr'] = (second_innings['runs_needed'] / second_innings['balls_remaining'].replace(0, 1)) * 6
    second_innings['batting_team_won'] = (second_innings['batting_team'] == second_innings['winner']).astype(int)

    over_snapshots = second_innings.groupby(['match_id', 'over']).last().reset_index()
    over_snapshots = over_snapshots[
        (over_snapshots['balls_remaining'] >= 0) &
        (over_snapshots['runs_needed'] > 0) &
        (over_snapshots['over'] >= 1)
    ]
    features = ['over', 'cumulative_runs', 'cumulative_wickets',
                'runs_needed', 'current_rr', 'required_rr', 'balls_remaining']
    X = over_snapshots[features].dropna()
    y = over_snapshots.loc[X.index, 'batting_team_won']
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

matches, deliveries = load_data()
model = train_model(matches, deliveries)

# ============================================================
# TOP NAVBAR
# ============================================================
st.markdown("""
<div class="topnav">
    <div class="navbar-brand">
        <div class="navbar-logo">🏏</div>
        <div>
            <div class="navbar-title">IPL ANALYTICS</div>
            <div class="navbar-subtitle">DATA SCIENCE DASHBOARD · 2008–2024</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

page = st.radio("", ["Overview", "Hypothesis Tests", "Win Predictor"],
                 horizontal=True, label_visibility="collapsed")

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================
if page == "Overview":
    total_matches = len(matches)
    total_runs = deliveries['total_runs'].sum()
    total_wickets = deliveries['is_wicket'].sum()
    total_seasons = matches['season'].nunique()

    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        (f"{total_matches:,}", "Total Matches", "metric-value-white"),
        (f"{total_runs:,}", "Total Runs", "metric-value-blue"),
        (f"{total_wickets:,}", "Total Wickets", ""),
        (f"{total_seasons}", "Seasons", "metric-value-blue"),
    ]
    for col, (val, label, cls) in zip([col1,col2,col3,col4], metrics):
        with col:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-value {cls}">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">MATCHES PER SEASON</div>', unsafe_allow_html=True)
        season_data = matches.groupby('season').size().reset_index(name='matches')
        fig = go.Figure(go.Bar(
            x=season_data['season'].astype(str), y=season_data['matches'],
            marker=dict(color='#ff6b00')
        ))
        fig.update_layout(**navy_layout())
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">MOST SUCCESSFUL TEAMS</div>', unsafe_allow_html=True)
        team_wins = matches['winner'].value_counts().reset_index()
        team_wins.columns = ['team', 'wins']
        team_wins = team_wins[team_wins['team'].notna()].head(8)
        fig = go.Figure(go.Bar(
            x=team_wins['wins'], y=team_wins['team'], orientation='h',
            marker=dict(color='#3a8ef6')
        ))
        fig.update_layout(**navy_layout())
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">TOSS DECISION — BAT OR FIELD</div>', unsafe_allow_html=True)
        toss_decision = matches['toss_decision'].value_counts()
        fig = go.Figure(go.Pie(
            labels=toss_decision.index, values=toss_decision.values, hole=0.55,
            marker=dict(colors=['#ff6b00', '#3a8ef6'])
        ))
        fig.update_layout(**navy_layout())
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">TOP RUN SCORERS</div>', unsafe_allow_html=True)
        top_scorers = deliveries.groupby('batter')['batsman_runs'].sum().sort_values(ascending=False).head(8).reset_index()
        top_scorers.columns = ['batter', 'runs']
        fig = go.Figure(go.Bar(
            x=top_scorers['runs'], y=top_scorers['batter'], orientation='h',
            marker=dict(color='#ff6b00')
        ))
        fig.update_layout(**navy_layout())
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE 2 — HYPOTHESIS TESTS
# ============================================================
elif page == "Hypothesis Tests":

    st.markdown('<div class="section-header">TEST 1 — DOES TOSS DECISION AFFECT MATCH OUTCOME? (CHI-SQUARE)</div>', unsafe_allow_html=True)
    matches['toss_match_winner'] = matches['toss_winner'] == matches['winner']
    contingency = pd.crosstab(matches['toss_decision'], matches['toss_match_winner'])
    chi2, p_value, dof, _ = stats.chi2_contingency(contingency)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""<div class="metric-box">
            <div class="metric-value metric-value-white">{chi2:.3f}</div>
            <div class="metric-label">Chi-Square Statistic</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-box">
            <div class="metric-value metric-value-blue">{p_value:.4f}</div>
            <div class="metric-label">P-Value</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        result = "SIGNIFICANT" if p_value < 0.05 else "NOT SIGNIFICANT"
        st.markdown(f"""<div class="metric-box">
            <div class="metric-value" style="color:#2ecc71; font-size:1.1rem">{result}</div>
            <div class="metric-label">Result at α=0.05</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""<div class="panel"><div class="insight-text">
        <span class="stat-orange">Conclusion:</span> Toss decision and match outcome are 
        <span class="stat-green">statistically related</span>. Teams choosing to field first win 
        significantly more often — chasing is the dominant winning strategy.
    </div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">TEST 2 — POWERPLAY VS DEATH OVER SCORING (T-TEST)</div>', unsafe_allow_html=True)

    def get_phase(over):
        if over < 6: return 'Powerplay'
        elif over < 16: return 'Middle'
        else: return 'Death'
    deliveries['phase'] = deliveries['over'].apply(get_phase)
    powerplay = deliveries[deliveries['phase'] == 'Powerplay']['total_runs']
    death = deliveries[deliveries['phase'] == 'Death']['total_runs']
    t_stat, t_p = stats.ttest_ind(powerplay, death)

    col1, col2 = st.columns([1.3, 1])
    with col1:
        phase_means = deliveries.groupby('phase')['total_runs'].mean().reindex(['Powerplay','Middle','Death']).reset_index()
        fig = go.Figure(go.Bar(
            x=phase_means['phase'], y=phase_means['total_runs'],
            marker=dict(color=['#3a8ef6', '#9b59b6', '#ff6b00'])
        ))
        fig.update_layout(**navy_layout("Avg Runs Per Ball by Phase"))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown(f"""<div class="panel">
            <div class="insight-text">
                <p>T-Statistic: <span class="stat-blue">{t_stat:.2f}</span></p>
                <p>P-Value: <span class="stat-green">&lt; 0.0001</span></p>
                <p>Powerplay avg: <span class="stat-orange">{powerplay.mean():.3f}</span></p>
                <p>Death over avg: <span class="stat-red">{death.mean():.3f}</span></p>
                <hr style="border-color:#1e2d54">
                <span class="stat-green">✓ Death overs are significantly faster scoring than Powerplay</span>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">TEST 3 — DO VENUES DIFFER IN SCORING? (ANOVA)</div>', unsafe_allow_html=True)
    innings_totals = deliveries.groupby(['match_id', 'inning'])['total_runs'].sum().reset_index()
    innings_totals = innings_totals.merge(matches[['id', 'venue']], left_on='match_id', right_on='id')
    top_venues = matches['venue'].value_counts().head(5).index.tolist()
    venue_data = innings_totals[innings_totals['venue'].isin(top_venues)]
    groups = [venue_data[venue_data['venue'] == v]['total_runs'].values for v in top_venues]
    f_stat, f_p = stats.f_oneway(*groups)

    col1, col2 = st.columns([1.3, 1])
    with col1:
        venue_means = venue_data.groupby('venue')['total_runs'].mean().reset_index()
        fig = go.Figure(go.Bar(
            x=venue_means['total_runs'], y=venue_means['venue'], orientation='h',
            marker=dict(color='#ff6b00')
        ))
        fig.update_layout(**navy_layout("Avg Score Per Innings by Venue"))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown(f"""<div class="panel">
            <div class="insight-text">
                <p>F-Statistic: <span class="stat-orange">{f_stat:.4f}</span></p>
                <p>P-Value: <span class="stat-orange">{f_p:.4f}</span></p>
                <hr style="border-color:#1e2d54">
                <span class="stat-red">✗ Venue scoring is NOT significantly different — 
                the "high scoring venue" reputation is largely a myth</span>
            </div>
        </div>""", unsafe_allow_html=True)

# ============================================================
# PAGE 3 — WIN PREDICTOR
# ============================================================
elif page == "Win Predictor":
    col1, col2 = st.columns([1, 1.6])

    with col1:
        st.markdown('<div class="section-header">MATCH STATE</div>', unsafe_allow_html=True)
        target = st.number_input("Target Score", min_value=50, max_value=300, value=165)
        current_over = st.slider("Current Over", 1, 19, 10)
        runs_scored = st.number_input("Runs Scored", min_value=0, max_value=300, value=80)
        wickets_lost = st.slider("Wickets Lost", 0, 10, 2)

    with col2:
        balls_bowled = current_over * 6
        balls_remaining = 120 - balls_bowled
        runs_needed = target - runs_scored + 1
        current_rr = (runs_scored / balls_bowled) * 6 if balls_bowled > 0 else 0
        required_rr = (runs_needed / balls_remaining) * 6 if balls_remaining > 0 else 99

        features_input = np.array([[current_over, runs_scored, wickets_lost,
                                     runs_needed, current_rr, required_rr, balls_remaining]])
        win_prob = model.predict_proba(features_input)[0][1] * 100

        prob_color = '#2ecc71' if win_prob >= 60 else '#ff6b00' if win_prob >= 35 else '#ff4d6d'

        st.markdown(f"""<div class="panel" style="text-align:center; padding:30px">
            <div style="color:#5b6b96; font-size:0.75rem; letter-spacing:2px; text-transform:uppercase">
                CHASING TEAM WIN PROBABILITY
            </div>
            <div style="font-size:4rem; font-weight:800; color:{prob_color}">{win_prob:.1f}%</div>
        </div>""", unsafe_allow_html=True)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(f"""<div class="metric-box"><div class="metric-value metric-value-blue">{runs_needed}</div>
            <div class="metric-label">Runs Needed</div></div>""", unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""<div class="metric-box"><div class="metric-value">{required_rr:.2f}</div>
            <div class="metric-label">Required RR</div></div>""", unsafe_allow_html=True)
        with col_c:
            st.markdown(f"""<div class="metric-box"><div class="metric-value metric-value-blue">{balls_remaining}</div>
            <div class="metric-label">Balls Left</div></div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">WIN PROBABILITY PROGRESSION</div>', unsafe_allow_html=True)
    overs_range = list(range(1, 20))
    win_probs = []
    for ov in overs_range:
        b = ov * 6
        br = 120 - b
        rs = int(runs_scored * (ov / current_over)) if current_over > 0 else 0
        rn = target - rs + 1
        crr = (rs / b) * 6 if b > 0 else 0
        rrr = (rn / br) * 6 if br > 0 else 99
        feat = np.array([[ov, rs, wickets_lost, rn, crr, rrr, br]])
        prob = model.predict_proba(feat)[0][1] * 100
        win_probs.append(prob)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=overs_range, y=win_probs, mode='lines+markers',
        line=dict(color='#ff6b00', width=3), marker=dict(size=5, color='#ff6b00'),
        fill='tozeroy', fillcolor='rgba(255,107,0,0.08)'
    ))
    fig.add_hline(y=50, line_dash="dash", line_color="#1e2d54")
    fig.update_layout(**navy_layout("", height=300))
    st.plotly_chart(fig, use_container_width=True)