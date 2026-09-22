import { useEffect, useMemo, useState } from "react";
import "./App.css";

const API = "/api";

const Icon = ({ children, size = 18 }) => (
  <span className="icon" style={{ width: size, height: size }} aria-hidden="true">{children}</span>
);

function Sparkline({ prices = [] }) {
  if (prices.length < 2) return <div className="empty-chart">Market data unavailable</div>;
  const min = Math.min(...prices);
  const max = Math.max(...prices);
  const range = max - min || 1;
  const points = prices.map((p, i) => {
    const x = (i / (prices.length - 1)) * 100;
    const y = 92 - ((p - min) / range) * 72;
    return `${x},${y}`;
  }).join(" ");
  const area = `0,100 ${points} 100,100`;
  return (
    <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="sparkline">
      <defs>
        <linearGradient id="goldFill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#ffc400" stopOpacity=".24" />
          <stop offset="1" stopColor="#ffc400" stopOpacity="0" />
        </linearGradient>
      </defs>
      {[25,50,75].map(y => <line key={y} x1="0" x2="100" y1={y} y2={y} className="chart-grid" />)}
      <polygon points={area} fill="url(#goldFill)" />
      <polyline points={points} className="chart-path" />
    </svg>
  );
}

function ConfidenceRing({ value }) {
  return (
    <div className="confidence-ring" style={{ "--value": `${value}%` }}>
      <div className="confidence-inner">{value.toFixed(2)}%</div>
    </div>
  );
}

function App() {
  const [dashboard, setDashboard] = useState(null);
  const [market, setMarket] = useState([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [pipelineRunning, setPipelineRunning] = useState(false);
  const [message, setMessage] = useState("");
  const [range, setRange] = useState("1M");

  const loadData = async (refresh = false) => {
    if (refresh) setRefreshing(true);
    try {
      const [d, m] = await Promise.all([fetch(`${API}/dashboard`), fetch(`${API}/market`)]);
      if (!d.ok) throw new Error("Backend unavailable");
      setDashboard(await d.json());
      setMarket(m.ok ? await m.json() : []);
      setMessage("");
    } catch (e) {
      console.error(e);
      setMessage("Backend connection unavailable.");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    loadData();
    const timer = setInterval(() => loadData(), 30000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    const nodes = document.querySelectorAll(".reveal");
    const observer = new IntersectionObserver(entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) entry.target.classList.add("show");
      });
    }, { threshold: .08 });
    nodes.forEach(n => observer.observe(n));
    return () => observer.disconnect();
  }, [loading, market]);

  const scrollTo = (id) => document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });

  const runPipeline = async () => {
    if (pipelineRunning) return;
    setPipelineRunning(true);
    setMessage("Starting daily pipeline...");
    try {
      const r = await fetch(`${API}/run-pipeline`, { method: "POST" });
      const result = await r.json();
      if (result.status === "already_running") {
        setMessage("Pipeline is already running.");
        setPipelineRunning(false);
        return;
      }
      setMessage("Pipeline running...");
      setTimeout(async () => {
        await loadData(true);
        setMessage("Dashboard updated.");
        setPipelineRunning(false);
      }, 6000);
    } catch (e) {
      console.error(e);
      setMessage("Could not start pipeline.");
      setPipelineRunning(false);
    }
  };

  const openUrl = (url) => window.open(url, "_blank", "noopener,noreferrer");
  const download = (kind) => window.open(`${API}/download/${kind}`, "_blank", "noopener,noreferrer");

  const prices = useMemo(() => {
    const all = market.map(x => Number(x.Close)).filter(Number.isFinite);
    const count = range === "1M" ? 22 : range === "3M" ? 66 : range === "6M" ? 132 : 180;
    return all.slice(-count);
  }, [market, range]);

  if (loading || !dashboard) {
    return (
      <div className="loading-screen">
        <div className="loading-mark">M</div>
        <strong>MURTAZA LABS</strong>
        <span>INITIALIZING MARKET INTELLIGENCE</span>
      </div>
    );
  }

  const isUp = String(dashboard.prediction).toUpperCase() === "UP";
  const confidence = Number(dashboard.confidence || 0) * 100;
  const up = Number(dashboard.probability_up || 0) * 100;
  const down = Number(dashboard.probability_down || 0) * 100;
  const accuracy = dashboard.accuracy == null ? "—" : `${Number(dashboard.accuracy).toFixed(0)}%`;
  const latest = Number(dashboard.nifty_close || 0).toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 });

  return (
    <div className="app">
      <header className="navbar">
        <button className="brand" onClick={() => scrollTo("home")}>
          <span className="bat-logo">🦇</span>
          <span className="brand-copy">
            <b>MURTAZA LABS</b>
            <small>DISCIPLINE · DATA · A BETTER TOMORROW</small>
          </span>
        </button>

        <nav className="nav-links">
          {[
            ["Home", "home"], ["Market", "dashboard"], ["Model", "model"], ["Research", "research"], ["About", "about"]
          ].map(([label, id]) => (
            <button key={id} onClick={() => scrollTo(id)} className={id === "home" ? "active" : ""}>{label}</button>
          ))}
        </nav>

        <div className="nav-right">
          <button className="social" onClick={() => openUrl("https://github.com/mohdmurtaza06")} aria-label="GitHub">◉</button>
          <button className="social linkedin" onClick={() => openUrl("https://www.linkedin.com/")} aria-label="LinkedIn">in</button>
          <span className="nav-divider" />
          <div className="quote">“IT’S NOT JUST DATA,<br />IT’S A HIGHER PURPOSE.”</div>
        </div>
      </header>

      <main>
        <section id="home" className="hero">
          <div className="hero-art" />
          <div className="hero-overlay" />
          <div className="hero-content">
            <h1>NIFTY 50<span>INTELLIGENCE</span></h1>
            <p className="hero-sub">MACHINE LEARNING · QUANTITATIVE RESEARCH · MARKET ANALYSIS</p>
            <p className="hero-quote">“SOME MEN JUST WANT TO WATCH THE MARKET BURN...<br />I WANT TO PREDICT IT.”<br /><b>— MOHAMMED MURTAZA TAJAMMUL</b></p>
            <div className="hero-actions">
              <button className="gold-btn" onClick={() => scrollTo("model")}>VIEW MODEL</button>
              <button className="outline-btn" onClick={() => scrollTo("research")}>EXPLORE RESEARCH</button>
            </div>
          </div>
          <div className="discipline">DISCIPLINE<br />BEATS<br />EMOTION.<br />ALWAYS.</div>
        </section>

        <section className="status-strip">
          <div><Icon>▣</Icon><span>PREDICTION DATE<strong>{dashboard.date}</strong></span></div>
          <div><Icon>↗</Icon><span>MARKET STATUS<strong>CLOSED</strong></span></div>
          <div><Icon>◷</Icon><span>LAST UPDATED<strong>{dashboard.date} · 04:12 PM</strong></span></div>
          <div className="online"><i /> <span>MODEL ONLINE<strong>RUNNING SMOOTHLY</strong></span></div>
        </section>

        <section id="dashboard" className="dashboard-section">
          {message && <div className="toast"><i />{message}</div>}

          <div className="overview-grid">
            <article className="dash-card price-card reveal">
              <div className="card-head"><span>NIFTY 50</span><Icon>▥</Icon></div>
              <strong>{latest}</strong>
              <small>Latest available market close</small>
              <div className="mini-chart"><Sparkline prices={prices.slice(-35)} /></div>
            </article>

            <article className={`dash-card prediction-card ${isUp ? "up-card" : "down-card"} reveal`}>
              <div className="card-head"><span>MODEL PREDICTION</span><Icon>{isUp ? "↗" : "◷"}</Icon></div>
              <div className="prediction-word"><span>{isUp ? "▲" : "▼"}</span>{isUp ? "UP" : "DOWN"}</div>
              <small>{isUp ? "BULLISH" : "BEARISH"}</small>
            </article>

            <article className="dash-card confidence-card reveal">
              <div className="card-head"><span>CONFIDENCE</span><Icon>♢</Icon></div>
              <ConfidenceRing value={confidence} />
              <small>Probability of predicted direction</small>
            </article>

            <article className="dash-card probability-card reveal">
              <div className="card-head"><span>PROBABILITY</span><Icon>▥</Icon></div>
              <div className="prob-row"><span>UP</span><b>{up.toFixed(2)}%</b></div>
              <div className="prob-track"><i className="green" style={{ width: `${up}%` }} /></div>
              <div className="prob-row"><span>DOWN</span><b>{down.toFixed(2)}%</b></div>
              <div className="prob-track"><i className="red" style={{ width: `${down}%` }} /></div>
            </article>
          </div>

          <div className="lower-grid">
            <article className="panel chart-panel reveal">
              <div className="panel-head">
                <span>NIFTY 50 PRICE CHART</span>
                <div className="range-tabs">
                  {["1M", "3M", "6M", "1Y"].map(r => <button key={r} className={range === r ? "selected" : ""} onClick={() => setRange(r)}>{r}</button>)}
                </div>
              </div>
              <div className="big-chart">
                <Sparkline prices={prices} />
                <div className="y-labels"><span>24,000</span><span>23,500</span><span>23,000</span><span>22,500</span><span>22,000</span></div>
              </div>
              <div className="x-labels"><span>Start</span><span>Mid</span><span>Latest</span></div>
            </article>

            <article className="panel tracking-panel reveal">
              <div className="panel-head"><span>LIVE MODEL TRACKING</span><Icon>◷</Icon></div>
              <div className="tracking-stats">
                {[
                  ["Total Predictions", dashboard.total_predictions],
                  ["Verified Predictions", dashboard.verified_predictions],
                  ["Correct Predictions", dashboard.correct],
                  ["Live Accuracy", accuracy]
                ].map(([label, value]) => <div key={label}><strong>{value}</strong><small>{label}</small></div>)}
              </div>
              <div className="tracking-note">
                <span className="note-bat">🦇</span>
                <div><b>“We don’t measure success in a day.”</b><p>Predictions are recorded and verified against future market data. Accuracy will become meaningful as more predictions are resolved.</p></div>
              </div>
              <button className="text-btn" onClick={() => download("history")}>DOWNLOAD HISTORY ↓</button>
            </article>
          </div>

          <div className="bottom-grid">
            <article id="performance" className="panel backtest-panel reveal">
              <div className="panel-head"><span>HISTORICAL BACKTEST PERFORMANCE</span><Icon>▥</Icon></div>
              <div className="bars">
                {[
                  ["Technical Model", "+13.94%", 29],
                  ["External Features", "+24.63%", 50],
                  ["Buy & Hold", "+48.81%", 100],
                  ["External Sharpe", "0.608", 34]
                ].map(([label, value, h]) => <div className="bar-item" key={label}><b>{value}</b><i style={{ height: `${h}%` }} /><span>{label}</span></div>)}
              </div>
            </article>

            <article id="research" className="panel research-panel reveal">
              <div className="panel-head"><span>RESEARCH & METHODOLOGY</span></div>
              <div className="method-grid">
                <div><Icon>⌘</Icon><span>XGBoost<small>Machine Learning</small></span></div>
                <div><Icon>▥</Icon><span>Technical Indicators<small>Price · Volume · Momentum</small></span></div>
                <div><Icon>⬡</Icon><span>External Market Features<small>VIX · Global Indices · Commodities</small></span></div>
                <div><Icon>✥</Icon><span>Walk-Forward Validation<small>Robust Out-of-Sample Testing</small></span></div>
              </div>
            </article>

            <article id="about" className="panel batman-panel reveal">
              <div className="portrait" />
              <div className="portrait-copy">A<br />DISCIPLINED<br />MIND<br />CREATES<br />A BRIGHTER<br />TOMORROW.</div>
            </article>
          </div>
        </section>
      </main>

      <footer>
        <div className="footer-brand"><span className="bat-logo">🦇</span><div><b>MURTAZA LABS</b><small>NIFTY 50 INTELLIGENCE</small></div></div>
        <div className="footer-center"><small>Built & Researched by</small><b>Mohammed Murtaza Tajammul</b><span>CSE · Muffakham Jah College of Engineering & Technology · Osmania University</span></div>
        <div className="footer-quote">“THE NIGHT IS DARKEST<br />JUST BEFORE THE OPEN.”</div>
      </footer>

      <button className="refresh-fab" onClick={() => loadData(true)} disabled={refreshing} title="Refresh data">{refreshing ? "…" : "↻"}</button>
    </div>
  );
}

export default App;
