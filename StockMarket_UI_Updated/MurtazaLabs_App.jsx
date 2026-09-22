import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

const API = "/api";

function Sparkline({ prices = [], large = false }) {
  if (prices.length < 2) {
    return <div className="empty-chart">MARKET DATA UNAVAILABLE</div>;
  }

  const values = prices.map(Number);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;

  const points = values
    .map((value, index) => {
      const x = (index / (values.length - 1)) * 100;
      const y = 92 - ((value - min) / range) * 75;
      return `${x},${y}`;
    })
    .join(" ");

  const last = points.split(" ").at(-1).split(",");

  return (
    <svg
      className={`sparkline ${large ? "large" : ""}`}
      viewBox="0 0 100 100"
      preserveAspectRatio="none"
    >
      <polygon
        className="chart-area"
        points={`0,100 ${points} 100,100`}
      />
      <polyline className="chart-line" points={points} />
      <circle
        className="chart-dot"
        cx={last[0]}
        cy={last[1]}
        r="1.7"
      />
    </svg>
  );
}


function calculateMarketIntelligence(rows = []) {
  const clean = rows
    .map(row => ({
      close: Number(row.Close),
      high: Number(row.High),
      low: Number(row.Low),
      volume: Number(row.Volume),
    }))
    .filter(row => Number.isFinite(row.close));

  if (clean.length < 5) {
    return {
      trend: "INSUFFICIENT DATA",
      momentum: "INSUFFICIENT DATA",
      rsi: null,
      volatility: null,
      volumeRatio: null,
      sma20: null,
      sma50: null,
    };
  }

  const closes = clean.map(row => row.close);

  const sma = (values, period) => {
    if (values.length < period) return null;
    const slice = values.slice(-period);
    return slice.reduce((sum, value) => sum + value, 0) / period;
  };

  const returns = [];
  for (let i = 1; i < closes.length; i += 1) {
    if (closes[i - 1] !== 0) {
      returns.push((closes[i] - closes[i - 1]) / closes[i - 1]);
    }
  }

  const period = Math.min(14, returns.length);
  const recentReturns = returns.slice(-period);

  const gains = recentReturns
    .filter(value => value > 0)
    .reduce((sum, value) => sum + value, 0);

  const losses = recentReturns
    .filter(value => value < 0)
    .reduce((sum, value) => sum + Math.abs(value), 0);

  let rsi = 50;

  if (losses === 0) {
    rsi = gains > 0 ? 100 : 50;
  } else {
    const rs = gains / losses;
    rsi = 100 - 100 / (1 + rs);
  }

  const sma20 = sma(closes, 20);
  const sma50 = sma(closes, 50);

  const lastClose = closes.at(-1);

  let trend = "NEUTRAL";

  if (sma20 !== null) {
    trend =
      lastClose > sma20
        ? "BULLISH"
        : lastClose < sma20
        ? "BEARISH"
        : "NEUTRAL";
  }

  if (sma20 !== null && sma50 !== null) {
    if (lastClose > sma20 && sma20 > sma50) {
      trend = "BULLISH";
    } else if (lastClose < sma20 && sma20 < sma50) {
      trend = "BEARISH";
    } else {
      trend = "MIXED";
    }
  }

  const momentumWindow = Math.min(5, closes.length - 1);
  const momentumStart = closes[closes.length - 1 - momentumWindow];
  const momentum =
    momentumStart !== 0
      ? ((lastClose - momentumStart) / momentumStart) * 100
      : 0;

  const volatilityWindow = Math.min(20, returns.length);
  const volatilityReturns = returns.slice(-volatilityWindow);

  const mean =
    volatilityReturns.length > 0
      ? volatilityReturns.reduce((sum, value) => sum + value, 0) /
        volatilityReturns.length
      : 0;

  const variance =
    volatilityReturns.length > 1
      ? volatilityReturns.reduce(
          (sum, value) => sum + (value - mean) ** 2,
          0
        ) / (volatilityReturns.length - 1)
      : 0;

  const volatility = Math.sqrt(variance) * Math.sqrt(252) * 100;

  const volumes = clean
    .map(row => row.volume)
    .filter(value => Number.isFinite(value) && value > 0);

  let volumeRatio = null;

  if (volumes.length >= 5) {
    const recentVolume = volumes.at(-1);
    const averageVolume =
      volumes.slice(-Math.min(20, volumes.length)).reduce(
        (sum, value) => sum + value,
        0
      ) / Math.min(20, volumes.length);

    if (averageVolume > 0) {
      volumeRatio = recentVolume / averageVolume;
    }
  }

  return {
    trend,
    momentum,
    rsi,
    volatility,
    volumeRatio,
    sma20,
    sma50,
  };
}

function SignalBar({ label, value, tone = "gold", suffix = "" }) {
  const safeValue = Number.isFinite(Number(value))
    ? Math.max(0, Math.min(100, Number(value)))
    : 0;

  return (
    <div className="signal-bar-row">
      <div className="signal-bar-label">
        <span>{label}</span>
        <b>
          {Number.isFinite(Number(value))
            ? `${Number(value).toFixed(1)}${suffix}`
            : "—"}
        </b>
      </div>

      <div className="signal-bar-track">
        <i
          className={`signal-bar-fill ${tone}`}
          style={{ width: `${safeValue}%` }}
        />
      </div>
    </div>
  );
}


function advancedMarketAnalytics(rows = []) {
  const clean = rows
    .map((row) => ({
      date: row.Date || row.date || "",
      open: Number(row.Open),
      high: Number(row.High),
      low: Number(row.Low),
      close: Number(row.Close),
      volume: Number(row.Volume),
    }))
    .filter((row) => Number.isFinite(row.close))
    .sort((a, b) => String(a.date).localeCompare(String(b.date)));

  const closes = clean.map((r) => r.close);
  const highs = clean.map((r) => r.high);
  const lows = clean.map((r) => r.low);
  const volumes = clean.map((r) => r.volume);

  const sma = (period) => {
    if (closes.length < period) return null;
    return closes.slice(-period).reduce((a, b) => a + b, 0) / period;
  };

  const ema = (period) => {
    if (closes.length < period) return null;
    const k = 2 / (period + 1);
    let value = closes.slice(0, period).reduce((a, b) => a + b, 0) / period;
    for (let i = period; i < closes.length; i += 1) {
      value = closes[i] * k + value * (1 - k);
    }
    return value;
  };

  const returns = [];
  for (let i = 1; i < closes.length; i += 1) {
    if (closes[i - 1] !== 0) returns.push((closes[i] / closes[i - 1]) - 1);
  }

  const rsi = (period) => {
    if (closes.length <= period) return null;
    const rs = returns.slice(-period);
    const gains = rs.filter((v) => v > 0).reduce((a, b) => a + b, 0) / period;
    const losses = rs.filter((v) => v < 0).reduce((a, b) => a + Math.abs(b), 0) / period;
    if (losses === 0) return gains > 0 ? 100 : 50;
    const strength = gains / losses;
    return 100 - 100 / (1 + strength);
  };

  const macdFast = ema(12);
  const macdSlow = ema(26);
  const macd = macdFast != null && macdSlow != null ? macdFast - macdSlow : null;

  const trueRanges = [];
  for (let i = 1; i < clean.length; i += 1) {
    const prevClose = clean[i - 1].close;
    trueRanges.push(Math.max(
      clean[i].high - clean[i].low,
      Math.abs(clean[i].high - prevClose),
      Math.abs(clean[i].low - prevClose)
    ));
  }
  const atr14 = trueRanges.length >= 14
    ? trueRanges.slice(-14).reduce((a, b) => a + b, 0) / 14
    : null;

  const lastClose = closes.at(-1) ?? null;
  const lastOpen = clean.at(-1)?.open ?? null;
  const lastHigh = clean.at(-1)?.high ?? null;
  const lastLow = clean.at(-1)?.low ?? null;
  const dayRangePct = lastClose && Number.isFinite(lastHigh) && Number.isFinite(lastLow)
    ? ((lastHigh - lastLow) / lastClose) * 100
    : null;

  const bbPeriod = Math.min(20, closes.length);
  const bbValues = closes.slice(-bbPeriod);
  const bbMean = bbValues.length ? bbValues.reduce((a, b) => a + b, 0) / bbValues.length : null;
  const bbStd = bbValues.length > 1
    ? Math.sqrt(bbValues.reduce((s, v) => s + (v - bbMean) ** 2, 0) / (bbValues.length - 1))
    : null;
  const bbUpper = bbMean != null && bbStd != null ? bbMean + 2 * bbStd : null;
  const bbLower = bbMean != null && bbStd != null ? bbMean - 2 * bbStd : null;
  const bbPosition = bbUpper != null && bbLower != null && bbUpper !== bbLower
    ? ((lastClose - bbLower) / (bbUpper - bbLower)) * 100
    : null;

  const stochPeriod = Math.min(14, clean.length);
  const recentHighs = highs.slice(-stochPeriod).filter(Number.isFinite);
  const recentLows = lows.slice(-stochPeriod).filter(Number.isFinite);
  const highest = recentHighs.length ? Math.max(...recentHighs) : null;
  const lowest = recentLows.length ? Math.min(...recentLows) : null;
  const stochastic = highest != null && lowest != null && highest !== lowest
    ? ((lastClose - lowest) / (highest - lowest)) * 100
    : null;

  const williams = stochastic != null ? stochastic - 100 : null;

  const roc = (period) => {
    if (closes.length <= period) return null;
    const base = closes[closes.length - 1 - period];
    return base ? ((lastClose - base) / base) * 100 : null;
  };

  const volumeWindow = volumes.slice(-20).filter((v) => Number.isFinite(v) && v > 0);
  const avgVolume = volumeWindow.length
    ? volumeWindow.reduce((a, b) => a + b, 0) / volumeWindow.length
    : null;
  const volumeRatio = avgVolume && Number.isFinite(volumes.at(-1))
    ? volumes.at(-1) / avgVolume
    : null;

  const weekHigh = highs.length ? Math.max(...highs.slice(-252)) : null;
  const weekLow = lows.length ? Math.min(...lows.slice(-252)) : null;
  const fromHighPct = weekHigh ? ((lastClose - weekHigh) / weekHigh) * 100 : null;
  const fromLowPct = weekLow ? ((lastClose - weekLow) / weekLow) * 100 : null;

  const recent = returns.slice(-20);
  const meanReturn = recent.length ? recent.reduce((a, b) => a + b, 0) / recent.length : 0;
  const variance = recent.length > 1
    ? recent.reduce((s, v) => s + (v - meanReturn) ** 2, 0) / (recent.length - 1)
    : 0;
  const annualizedVol = Math.sqrt(variance) * Math.sqrt(252) * 100;

  const sma20 = sma(20);
  const sma50 = sma(50);
  const sma200 = sma(200);
  const ema20 = ema(20);

  let regime = "NEUTRAL";
  if (lastClose != null && sma20 != null && sma50 != null) {
    if (lastClose > sma20 && sma20 > sma50) regime = "BULLISH";
    else if (lastClose < sma20 && sma20 < sma50) regime = "BEARISH";
    else regime = "MIXED";
  }

  return {
    lastClose,
    lastOpen,
    lastHigh,
    lastLow,
    dayRangePct,
    sma20,
    sma50,
    sma200,
    ema20,
    rsi5: rsi(5),
    rsi14: rsi(14),
    rsi21: rsi(21),
    macd,
    atr14,
    bbUpper,
    bbLower,
    bbPosition,
    stochastic,
    williams,
    roc5: roc(5),
    roc20: roc(20),
    volumeRatio,
    annualizedVol,
    weekHigh,
    weekLow,
    fromHighPct,
    fromLowPct,
    regime,
    dataPoints: clean.length,
  };
}

function formatNumber(value, digits = 2) {
  const n = Number(value);
  return Number.isFinite(n)
    ? n.toLocaleString("en-IN", {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
      })
    : "—";
}

function formatPercent(value, digits = 2) {
  const n = Number(value);
  return Number.isFinite(n) ? `${n.toFixed(digits)}%` : "—";
}

function directionClass(value) {
  const n = Number(value);
  if (!Number.isFinite(n) || n === 0) return "neutral-text";
  return n > 0 ? "green-text" : "red-text";
}

function RegimePill({ label, value, tone = "gold" }) {
  return (
    <div className={`regime-pill ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function SectionTitle({ eyebrow, title, description }) {
  return (
    <div className="section-title reveal">
      <span>{eyebrow}</span>
      <h2>{title}</h2>
      <p>{description}</p>
    </div>
  );
}

function ConfidenceRing({ value }) {
  return (
    <div
      className="confidence-ring"
      style={{ "--confidence": `${value}%` }}
    >
      <div>
        <strong>{value.toFixed(2)}%</strong>
        <small>CONFIDENCE</small>
      </div>
    </div>
  );
}

function IntelligenceVisual() {
  return (
    <div className="intelligence-panel reveal">
      <div className="intelligence-grid" />

      <div className="signal-orb">
        <div className="orb-core">
          <span>ML</span>
        </div>

        <div className="orb-ring ring-one" />
        <div className="orb-ring ring-two" />
        <div className="orb-ring ring-three" />
      </div>

      <div className="signal-lines">
        <span />
        <span />
        <span />
        <span />
      </div>

      <div className="market-bars">
        {[32, 48, 37, 62, 55, 78, 68, 91, 74, 100].map(
          (height, i) => (
            <i key={i} style={{ height: `${height}%` }} />
          )
        )}
      </div>

      <div className="intelligence-label">
        <span>07 · MURTAZA LABS</span>

        <strong>
          MARKET
          <br />
          INTELLIGENCE
        </strong>

        <small>
          MACHINE LEARNING
          <br />
          QUANTITATIVE RESEARCH
          <br />
          WALK-FORWARD VALIDATION
        </small>
      </div>

      <div className="system-status">
        <i />
        MODEL ENGINE
        <b>ONLINE</b>
      </div>
    </div>
  );
}

export default function App() {
  const [dashboard, setDashboard] = useState(null);
  const [market, setMarket] = useState([]);
  const [history, setHistory] = useState([]);
  const [range, setRange] = useState("1M");
  const [active, setActive] = useState("home");
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [pipelineRunning, setPipelineRunning] = useState(false);
  const [pipelineStatus, setPipelineStatus] = useState(null);
  const [showTop, setShowTop] = useState(false);
  const [message, setMessage] = useState("");
  const [newPrediction, setNewPrediction] = useState(false);
  const [liveClock, setLiveClock] = useState(new Date());
  const lastPredictionKeyRef = useRef(null);
  const newPredictionTimerRef = useRef(null);

  useEffect(() => {
    const clock = setInterval(() => setLiveClock(new Date()), 1000);
    return () => clearInterval(clock);
  }, []);

  const loadData = async (refresh = false) => {
    try {
      if (refresh) setRefreshing(true);

      const [dashboardRes, marketRes, historyRes, pipelineRes] =
        await Promise.all([
          fetch(`${API}/dashboard`),
          fetch(`${API}/market`),
          fetch(`${API}/history`),
          fetch(`${API}/pipeline-status`),
        ]);

      if (!dashboardRes.ok) {
        throw new Error("Backend unavailable");
      }

      const nextDashboard = await dashboardRes.json();

      // Detect a genuinely new model prediction.
      // The ref avoids stale state inside the 30-second polling loop.
      const predictionKey = [
        nextDashboard.date,
        nextDashboard.prediction,
        nextDashboard.nifty_close,
        nextDashboard.confidence,
      ].join("|");

      if (lastPredictionKeyRef.current === null) {
        lastPredictionKeyRef.current = predictionKey;
      } else if (
        lastPredictionKeyRef.current !== predictionKey
      ) {
        lastPredictionKeyRef.current = predictionKey;
        setNewPrediction(true);
        showMessage("NEW MARKET PREDICTION AVAILABLE.");

        if (newPredictionTimerRef.current) {
          clearTimeout(newPredictionTimerRef.current);
        }

        newPredictionTimerRef.current = setTimeout(() => {
          setNewPrediction(false);
        }, 6500);
      }

      setDashboard(nextDashboard);
      setMarket(
        marketRes.ok ? await marketRes.json() : []
      );
      setHistory(
        historyRes.ok ? await historyRes.json() : []
      );

      if (pipelineRes.ok) {
        setPipelineStatus(await pipelineRes.json());
      }
    } catch (error) {
      console.error(error);
      setMessage("BACKEND CONNECTION UNAVAILABLE");
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    loadData();

    const timer = setInterval(() => {
      loadData();
    }, 30000);

    return () => {
      clearInterval(timer);

      if (newPredictionTimerRef.current) {
        clearTimeout(newPredictionTimerRef.current);
      }
    };
  }, []);

  useEffect(() => {
    const observer = new IntersectionObserver(
      entries => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            entry.target.classList.add("show");
          }
        });
      },
      { threshold: 0.08 }
    );

    document
      .querySelectorAll(".reveal")
      .forEach(el => observer.observe(el));

    return () => observer.disconnect();
  }, [loading, market]);

  useEffect(() => {
    const sections = document.querySelectorAll(
      "main section[id]"
    );

    const observer = new IntersectionObserver(
      entries => {
        const visible = entries
          .filter(entry => entry.isIntersecting)
          .sort(
            (a, b) =>
              b.intersectionRatio -
              a.intersectionRatio
          )[0];

        if (visible) {
          setActive(visible.target.id);
        }
      },
      {
        threshold: [0.2, 0.4, 0.6],
      }
    );

    sections.forEach(section =>
      observer.observe(section)
    );

    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const handleScroll = () => {
      setShowTop(window.scrollY > 700);

      const height =
        document.documentElement.scrollHeight -
        window.innerHeight;

      const progress =
        height > 0
          ? (window.scrollY / height) * 100
          : 0;

      document.documentElement.style.setProperty(
        "--scroll-progress",
        `${progress}%`
      );
    };

    window.addEventListener(
      "scroll",
      handleScroll
    );

    return () =>
      window.removeEventListener(
        "scroll",
        handleScroll
      );
  }, []);

  useEffect(() => {
    const handleMouse = event => {
      document.documentElement.style.setProperty(
        "--mouse-x",
        `${event.clientX}px`
      );

      document.documentElement.style.setProperty(
        "--mouse-y",
        `${event.clientY}px`
      );
    };

    window.addEventListener(
      "mousemove",
      handleMouse
    );

    return () =>
      window.removeEventListener(
        "mousemove",
        handleMouse
      );
  }, []);

  const scrollTo = id => {
    document.getElementById(id)?.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  };

  const openExternal = url => {
    window.open(
      url,
      "_blank",
      "noopener,noreferrer"
    );
  };

  const download = endpoint => {
    window.open(
      `${API}/${endpoint}`,
      "_blank",
      "noopener,noreferrer"
    );
  };

  const showMessage = text => {
    setMessage(text);

    setTimeout(() => {
      setMessage("");
    }, 3500);
  };

  const runPipeline = async () => {
    if (pipelineRunning) return;

    setPipelineRunning(true);
    showMessage("STARTING PREDICTION PIPELINE...");

    try {
      const response = await fetch(
        `${API}/run-pipeline`,
        {
          method: "POST",
        }
      );

      const result = await response.json();

      if (result.status === "already_running") {
        showMessage("PIPELINE IS ALREADY RUNNING.");
        setPipelineRunning(false);
        return;
      }

      showMessage(
        "PIPELINE STARTED. UPDATING DATA..."
      );

      let checks = 0;

      const checkPipeline = async () => {
        checks += 1;

        try {
          const statusResponse = await fetch(
            `${API}/pipeline-status`
          );

          if (statusResponse.ok) {
            const status = await statusResponse.json();

            setPipelineStatus(status);

            if (!status.running) {
              await loadData(true);

              if (
                String(status.last_status || "").toLowerCase() ===
                "failed"
              ) {
                showMessage(
                  "PIPELINE FAILED. CHECK BACKEND LOGS."
                );
              } else {
                showMessage(
                  "PREDICTION PIPELINE COMPLETED."
                );
              }

              setPipelineRunning(false);
              return;
            }
          }
        } catch (error) {
          console.error(error);
        }

        if (checks < 60) {
          setTimeout(checkPipeline, 3000);
        } else {
          setPipelineRunning(false);
          showMessage(
            "PIPELINE IS STILL RUNNING. STATUS WILL CONTINUE UPDATING."
          );
        }
      };

      setTimeout(checkPipeline, 2000);
    } catch (error) {
      console.error(error);
      showMessage(
        "UNABLE TO START PREDICTION PIPELINE."
      );
      setPipelineRunning(false);
    }
  };

  const chartRows = useMemo(() => {
    const counts = {
      "1M": 22,
      "3M": 66,
      "6M": 132,
      "1Y": 250,
    };

    return market
      .filter(row =>
        Number.isFinite(Number(row.Close))
      )
      .slice(-(counts[range] || 22));
  }, [market, range]);

  const prices = chartRows.map(row =>
    Number(row.Close)
  );

  const intelligence = useMemo(
    () => calculateMarketIntelligence(market),
    [market]
  );

  const advanced = useMemo(
    () => advancedMarketAnalytics(market),
    [market]
  );

  const dayChangePct = Number(dashboard?.day_change_pct || 0) * 100;
  const productionRSI = Number(dashboard?.rsi_14);
  const productionTrend = Number(dashboard?.trend_regime);
  const productionVolatility = Number(dashboard?.volatility_regime);
  const productionRisk = Number(dashboard?.risk_score);

  const volatilityLabel = Number.isFinite(productionVolatility)
    ? productionVolatility >= 1.1
      ? "HIGH"
      : productionVolatility >= 0.65
      ? "NORMAL"
      : "LOW"
    : "—";

  const riskLabel = Number.isFinite(productionRisk)
    ? productionRisk > 0.15
      ? "HIGH"
      : productionRisk < -0.15
      ? "LOW"
      : "MODERATE"
    : "—";

  const marketSession = (() => {
    const formatter = new Intl.DateTimeFormat("en-IN", {
      timeZone: "Asia/Kolkata",
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    });
    const time = formatter.format(liveClock);
    const [h, m] = time.split(":").map(Number);
    const minutes = h * 60 + m;
    if (minutes >= 555 && minutes <= 930) return "MARKET HOURS";
    return "MARKET CLOSED";
  })();

  const rsiTone =
    intelligence.rsi == null
      ? "gold"
      : intelligence.rsi >= 70
      ? "red"
      : intelligence.rsi <= 30
      ? "green"
      : "gold";

  const rsiBar =
    intelligence.rsi == null
      ? 0
      : intelligence.rsi;

  const momentumBar =
    intelligence.momentum == null
      ? 50
      : Math.max(
          0,
          Math.min(
            100,
            50 + intelligence.momentum * 10
          )
        );

  const volatilityBar =
    intelligence.volatility == null
      ? 0
      : Math.max(
          0,
          Math.min(
            100,
            intelligence.volatility * 3
          )
        );

  const volumeBar =
    intelligence.volumeRatio == null
      ? 0
      : Math.max(
          0,
          Math.min(
            100,
            intelligence.volumeRatio * 50
          )
        );

  if (loading || !dashboard) {
    return (
      <div className="loading-screen">
        <div className="loading-mark">ML</div>

        <div>
          <strong>MURTAZA LABS</strong>
          <span>
            INITIALIZING NIFTY INTELLIGENCE
          </span>
        </div>
      </div>
    );
  }

  const prediction = String(
    dashboard.prediction || "DOWN"
  ).toUpperCase();

  const isUp = prediction === "UP";

  const asPercent = value => {
    const number = Number(value);
    if (!Number.isFinite(number)) return 0;
    return Math.abs(number) <= 1 ? number * 100 : number;
  };

  const confidence = asPercent(dashboard.confidence);
  const probabilityUp = asPercent(dashboard.probability_up);
  const probabilityDown = asPercent(dashboard.probability_down);

  const nifty = Number(
    dashboard.nifty_close || 0
  ).toLocaleString("en-IN", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });

  const totalPredictions = Number(
    dashboard.total_predictions ??
      history.length
  );

  const verified = Number(
    dashboard.verified_predictions ?? 0
  );

  const correct = Number(
    dashboard.correct ?? 0
  );

  const accuracy =
    verified > 0
      ? (correct / verified) * 100
      : null;

  return (
    <div className="app">
      <div className="scroll-progress" />

      {/* NAVBAR */}

      <header className="navbar">
        <button
          className="brand"
          onClick={() => scrollTo("home")}
        >
          <span className="lab-mark">ML</span>

          <span className="brand-copy">
            <b>MURTAZA LABS</b>

            <small>
              DISCIPLINE · DATA · A BETTER TOMORROW
            </small>
          </span>
        </button>

        <nav className="nav-links">
          {[
            ["HOME", "home"],
            ["COMMAND", "command"],
            ["MARKET", "market"],
            ["TECHNICAL", "technical"],
            ["MODEL", "model"],
            ["RESEARCH", "research"],
            ["ABOUT", "about"],
          ].map(([label, id]) => (
            <button
              key={id}
              className={
                active === id ? "active" : ""
              }
              onClick={() => scrollTo(id)}
            >
              {label}
            </button>
          ))}
        </nav>

        <div className="nav-actions">
          <button
            onClick={() =>
              openExternal(
                "https://github.com/mohdmurtaza06"
              )
            }
          >
            GH
          </button>

          <button
            onClick={() =>
              openExternal(
                "https://www.linkedin.com/"
              )
            }
          >
            in
          </button>
        </div>
      </header>

      <main>
        {/* HERO */}

        <section
          id="home"
          className="hero-section"
        >
          <div className="hero-background" />
          <div className="hero-vignette" />

          <div className="hero-content">
            <div className="hero-kicker">
              MURTAZA LABS · QUANTITATIVE INTELLIGENCE
            </div>

            <h1>
              NIFTY 50
              <span>INTELLIGENCE</span>
            </h1>

            <p className="hero-description">
              MACHINE LEARNING · QUANTITATIVE RESEARCH ·
              MARKET ANALYSIS
            </p>

            <div className="hero-statement">
              <span>
                DATA CHANGES. MARKETS CHANGE.
              </span>

              <span>
                THE SYSTEM KEEPS LEARNING.
              </span>

              <b>
                RESEARCH · VALIDATION · DISCIPLINE
              </b>
            </div>

            <div className="hero-buttons">
              <button
                className="gold-button"
                onClick={() =>
                  scrollTo("model")
                }
              >
                VIEW MODEL
              </button>

              <button
                className="outline-button"
                onClick={() =>
                  scrollTo("research")
                }
              >
                EXPLORE RESEARCH
              </button>
            </div>
          </div>

          <div className="hero-side-text">
            <span>DISCIPLINE</span>
            <span>BEATS</span>
            <span>EMOTION.</span>
          </div>

          <div className="hero-scroll">
            <span />
            SCROLL TO EXPLORE
          </div>
        </section>

        {/* STATUS */}

        <section className="status-strip">
          <div>
            <span>
              PREDICTION
              <strong>
                {dashboard.prediction || "—"}
              </strong>
            </span>
          </div>

          <div>
            <span>
              NIFTY 50
              <strong>{nifty}</strong>
            </span>
          </div>

          <div>
            <span>
              CONFIDENCE
              <strong>
                {confidence.toFixed(2)}%
              </strong>
            </span>
          </div>

          <div className="online">
            <i />

            <span>
              MODEL
              <strong>ONLINE</strong>
            </span>
          </div>
        </section>

        {/* COMMAND CENTER */}
        <section id="command" className="command-center-section">
          <div className="command-header reveal">
            <div>
              <span>LIVE COMMAND CENTER</span>
              <h2>THE MARKET AT A GLANCE.</h2>
              <p>Production model output, technical state, market regime and cross-market context in one screen.</p>
            </div>
            <div className="command-time">
              <span>{marketSession}</span>
              <strong>
                {liveClock.toLocaleTimeString("en-IN", {
                  timeZone: "Asia/Kolkata",
                  hour: "2-digit",
                  minute: "2-digit",
                  second: "2-digit",
                  hour12: false,
                })}
              </strong>
              <small>IST · {dashboard.date || "—"}</small>
            </div>
          </div>

          <div className="command-grid">
            <article className="command-hero-card reveal">
              <div className="command-card-top">
                <span>PRODUCTION MODEL</span>
                <b className={isUp ? "green-text" : "red-text"}>
                  {isUp ? "▲ UP" : "▼ DOWN"}
                </b>
              </div>
              <div className="command-price">{nifty}</div>
              <div className={`command-change ${directionClass(dayChangePct)}`}>
                {dayChangePct >= 0 ? "+" : ""}{dayChangePct.toFixed(2)}% DAY CHANGE
              </div>
              <div className="command-probabilities">
                <div>
                  <span>UP</span>
                  <strong>{probabilityUp.toFixed(2)}%</strong>
                  <i><b style={{ width: `${probabilityUp}%` }} /></i>
                </div>
                <div>
                  <span>DOWN</span>
                  <strong>{probabilityDown.toFixed(2)}%</strong>
                  <i><b className="down-fill" style={{ width: `${probabilityDown}%` }} /></i>
                </div>
              </div>
            </article>

            <article className="command-card reveal">
              <span>MODEL STATE</span>
              <div className="command-stat-large">{confidence.toFixed(2)}%</div>
              <small>Prediction confidence</small>
              <div className="command-mini-grid">
                <div><b>{dashboard.feature_count || 298}</b><span>FEATURES</span></div>
                <div><b>{dashboard.production_model || "CATBOOST"}</b><span>ENGINE</span></div>
              </div>
            </article>

            <article className="command-card reveal">
              <span>MARKET REGIME</span>
              <RegimePill
                label="PRICE STRUCTURE"
                value={advanced.regime}
                tone={advanced.regime === "BULLISH" ? "green" : advanced.regime === "BEARISH" ? "red" : "gold"}
              />
              <RegimePill
                label="MODEL TREND"
                value={Number.isFinite(productionTrend) ? `${productionTrend > 0 ? "+" : ""}${productionTrend.toFixed(0)}` : "—"}
                tone={productionTrend > 0 ? "green" : productionTrend < 0 ? "red" : "gold"}
              />
              <RegimePill label="VOLATILITY" value={volatilityLabel} tone={volatilityLabel === "HIGH" ? "red" : volatilityLabel === "LOW" ? "green" : "gold"} />
              <RegimePill label="RISK" value={riskLabel} tone={riskLabel === "HIGH" ? "red" : riskLabel === "LOW" ? "green" : "gold"} />
            </article>
          </div>

          <div className="command-lower-grid">
            <article className="command-panel reveal">
              <div className="command-panel-title">
                <span>TECHNICAL SNAPSHOT</span>
                <small>LIVE CALCULATION</small>
              </div>
              <div className="technical-matrix">
                {[
                  ["RSI 5", advanced.rsi5, "number", ""],
                  ["RSI 14", Number.isFinite(productionRSI) ? productionRSI : advanced.rsi14, "number", ""],
                  ["RSI 21", advanced.rsi21, "number", ""],
                  ["MACD", advanced.macd, "number", ""],
                  ["SMA 20", advanced.sma20, "price", ""],
                  ["SMA 50", advanced.sma50, "price", ""],
                  ["SMA 200", advanced.sma200, "price", ""],
                  ["ATR 14", advanced.atr14, "number", ""],
                  ["BB POSITION", advanced.bbPosition, "percent", ""],
                  ["STOCHASTIC", advanced.stochastic, "number", ""],
                  ["WILLIAMS %R", advanced.williams, "number", ""],
                  ["ROC 5D", advanced.roc5, "percent", ""],
                  ["ROC 20D", advanced.roc20, "percent", ""],
                  ["VOLUME RATIO", advanced.volumeRatio, "ratio", ""],
                ].map(([label, value, type]) => {
                  let display = "—";
                  if (Number.isFinite(Number(value))) {
                    if (type === "price") display = formatNumber(value);
                    else if (type === "percent") display = formatPercent(value);
                    else if (type === "ratio") display = `${Number(value).toFixed(2)}x`;
                    else display = Number(value).toFixed(2);
                  }
                  return (
                    <div className="technical-cell" key={label}>
                      <span>{label}</span>
                      <strong className={label.includes("ROC") || label === "MACD" ? directionClass(value) : ""}>{display}</strong>
                    </div>
                  );
                })}
              </div>
            </article>

            <article className="command-panel reveal">
              <div className="command-panel-title">
                <span>CROSS-MARKET</span>
                <small>PRODUCTION INPUTS</small>
              </div>
              <div className="cross-market-grid">
                {[
                  ["INDIA VIX", dashboard.india_vix, ""],
                  ["BANK NIFTY", dashboard.banknifty_return_1d, "%"],
                  ["S&P 500", dashboard.sp500_return_1d, "%"],
                  ["USD / INR", dashboard.usdinr_return_1d, "%"],
                  ["GOLD", dashboard.gold_return_1d, "%"],
                  ["CRUDE OIL", dashboard.crude_return_1d, "%"],
                ].map(([label, value, suffix]) => (
                  <div className="cross-market-cell" key={label}>
                    <span>{label}</span>
                    <strong className={suffix === "%" ? directionClass(Number(value)) : ""}>
                      {Number.isFinite(Number(value)) ? `${(Number(value) * (suffix === "%" ? 100 : 1)).toFixed(2)}${suffix}` : "—"}
                    </strong>
                  </div>
                ))}
              </div>
              <div className="cross-market-note">
                <i /> Cross-market features are read from the production prediction payload.
              </div>
            </article>
          </div>

          <div className="command-lower-grid second-row">
            <article className="command-panel reveal">
              <div className="command-panel-title">
                <span>PRICE POSITION</span>
                <small>252 SESSION RANGE</small>
              </div>
              <div className="range-summary">
                <div><span>52W HIGH</span><strong>{formatNumber(advanced.weekHigh)}</strong></div>
                <div><span>52W LOW</span><strong>{formatNumber(advanced.weekLow)}</strong></div>
                <div><span>FROM HIGH</span><strong className="red-text">{formatPercent(advanced.fromHighPct)}</strong></div>
                <div><span>FROM LOW</span><strong className="green-text">+{formatPercent(advanced.fromLowPct)}</strong></div>
              </div>
              <div className="range-track">
                <i />
              </div>
            </article>

            <article className="command-panel reveal">
              <div className="command-panel-title">
                <span>MODEL HEALTH</span>
                <small>PRODUCTION</small>
              </div>
              <div className="health-list">
                {[
                  ["MODEL ENGINE", dashboard.production_model || "CATBOOST", true],
                  ["FEATURE COUNT", `${dashboard.feature_count || 298}`, true],
                  ["TRAINED THROUGH", dashboard.model_trained_through || "—", true],
                  ["DATA POINTS", `${advanced.dataPoints}`, true],
                  ["PREDICTIONS", `${totalPredictions}`, true],
                  ["VERIFIED", `${verified}`, verified >= 0],
                ].map(([label, value, online]) => (
                  <div key={label}>
                    <span><i className={online ? "health-dot" : "health-dot error"} />{label}</span>
                    <strong>{value}</strong>
                  </div>
                ))}
              </div>
            </article>
          </div>
        </section>

        {/* PIPELINE STATUS */}

        <section className="pipeline-status-panel">
          <div className="pipeline-status-main">
            <span>AUTOMATION</span>

            <strong>
              <i
                className={
                  pipelineStatus?.running
                    ? "pipeline-dot running"
                    : pipelineStatus?.last_status === "failed"
                    ? "pipeline-dot error"
                    : "pipeline-dot online"
                }
              />

              {pipelineStatus?.running
                ? "RUNNING"
                : pipelineStatus?.last_status === "failed"
                ? "ERROR"
                : "ONLINE"}
            </strong>

            <small>
              Automatic daily ML pipeline
            </small>
          </div>

          <div>
            <span>LAST STARTED</span>

            <strong>
              {pipelineStatus?.last_started
                ? new Date(
                    pipelineStatus.last_started
                  ).toLocaleString("en-IN", {
                    timeZone: "Asia/Kolkata",
                    day: "2-digit",
                    month: "short",
                    year: "numeric",
                    hour: "2-digit",
                    minute: "2-digit",
                    hour12: false,
                  }) + " IST"
                : "—"}
            </strong>
          </div>

          <div>
            <span>LAST COMPLETED</span>

            <strong>
              {pipelineStatus?.last_completed
                ? new Date(
                    pipelineStatus.last_completed
                  ).toLocaleString("en-IN", {
                    timeZone: "Asia/Kolkata",
                    day: "2-digit",
                    month: "short",
                    year: "numeric",
                    hour: "2-digit",
                    minute: "2-digit",
                    hour12: false,
                  }) + " IST"
                : "—"}
            </strong>
          </div>

          <div>
            <span>SCHEDULER</span>

            <strong>
              15:45 IST
            </strong>

            <small>
              TRADING DAYS
            </small>
          </div>
        </section>

        {/* MARKET */}

        <section
          id="market"
          className="section"
        >
          <SectionTitle
            eyebrow="01 · MARKET INTELLIGENCE"
            title="THE MARKET, DISTILLED."
            description="A live intelligence layer combining price action, technical indicators, external signals and machine-learning predictions."
          />

          <div className="market-grid">
            <article className="metric-card reveal">
              <span>NIFTY 50</span>

              <strong>{nifty}</strong>

              <small>
                Latest market close
              </small>

              <div className="mini-chart">
                <Sparkline
                  prices={prices.slice(-35)}
                />
              </div>
            </article>

            <article
              className={`metric-card prediction ${
                isUp
                  ? "prediction-up"
                  : "prediction-down"
              } ${newPrediction ? "prediction-updated" : ""} reveal`}
            >
              <span>MODEL PREDICTION</span>

              {newPrediction && (
                <b className="new-prediction-badge">
                  NEW PREDICTION
                </b>
              )}

              <strong>
                {isUp ? "▲ UP" : "▼ DOWN"}
              </strong>

              <small>
                {isUp
                  ? "BULLISH SIGNAL"
                  : "BEARISH SIGNAL"}
              </small>

              <div className="prediction-line">
                LIVE DIRECTIONAL SIGNAL
              </div>
            </article>

            <article className="metric-card reveal">
              <span>CONFIDENCE</span>

              <ConfidenceRing
                value={confidence}
              />
            </article>

            <article className="metric-card reveal">
              <span>PROBABILITY</span>

              <div className="probability">
                <div>
                  <label>
                    UP
                    <b>
                      {probabilityUp.toFixed(2)}%
                    </b>
                  </label>

                  <div>
                    <i
                      style={{
                        width: `${probabilityUp}%`,
                      }}
                    />
                  </div>
                </div>

                <div>
                  <label>
                    DOWN
                    <b>
                      {probabilityDown.toFixed(2)}%
                    </b>
                  </label>

                  <div>
                    <i
                      className="red"
                      style={{
                        width: `${probabilityDown}%`,
                      }}
                    />
                  </div>
                </div>
              </div>
            </article>
          </div>

          <div className="market-chart-panel reveal">
            <div className="panel-header">
              <div>
                <span>
                  NIFTY 50 PRICE HISTORY
                </span>

                <small>
                  Historical market movement
                </small>
              </div>

              <div className="range-buttons">
                {[
                  "1M",
                  "3M",
                  "6M",
                  "1Y",
                ].map(item => (
                  <button
                    key={item}
                    className={
                      range === item
                        ? "selected"
                        : ""
                    }
                    onClick={() =>
                      setRange(item)
                    }
                  >
                    {item}
                  </button>
                ))}
              </div>
            </div>

            <div className="large-chart">
              <Sparkline
                prices={prices}
                large
              />

              <div className="chart-labels">
                <span>
                  {chartRows[0]?.Date ||
                    chartRows[0]?.date ||
                    ""}
                </span>

                <span>
                  {chartRows.at(-1)?.Date ||
                    chartRows.at(-1)?.date ||
                    ""}
                </span>
              </div>
            </div>
          </div>
        </section>

        {/* MODEL */}

        <section
          id="model"
          className="section dark-section"
        >
          <SectionTitle
            eyebrow="02 · MODEL INTELLIGENCE"
            title="INSIDE THE ENGINE."
            description="The prediction layer transforms market observations into engineered features before classification."
          />

          <div className="model-layout">
            <div className="model-main reveal">
              <div className="model-status">
                <i />
                MODEL ONLINE
              </div>

              <h3>
                CATBOOST
                <span>
                  PRODUCTION CLASSIFICATION ENGINE
                </span>
              </h3>

              <p>
                The system combines historical
                market data, technical indicators
                and external features before
                producing a directional prediction.
              </p>

              <div className="model-pipeline">
                {[
                  [
                    "01",
                    "Market Data",
                    "OHLCV & Returns",
                  ],
                  [
                    "02",
                    "Features",
                    "Momentum · Volatility",
                  ],
                  [
                    "03",
                    "External",
                    "Market Signals",
                  ],
                  [
                    "04",
                    "CatBoost",
                    "Production Classification",
                  ],
                  [
                    "05",
                    "Prediction",
                    "UP / DOWN",
                  ],
                ].map(
                  ([number, title, text]) => (
                    <button
                      key={number}
                      onClick={() =>
                        showMessage(
                          `${title}: ${text}`
                        )
                      }
                    >
                      <b>{number}</b>

                      <span>
                        {title}

                        <small>
                          {text}
                        </small>
                      </span>
                    </button>
                  )
                )}
              </div>
            </div>

            <div className="model-side">
              <div className="model-stat reveal">
                <span>
                  CURRENT SIGNAL
                </span>

                <strong
                  className={
                    isUp
                      ? "green-text"
                      : "red-text"
                  }
                >
                  {prediction}
                </strong>

                <small>
                  Current model direction
                </small>
              </div>

              <div className="model-stat reveal">
                <span>CONFIDENCE</span>

                <strong>
                  {confidence.toFixed(2)}%
                </strong>

                <small>
                  Model confidence
                </small>
              </div>

              <div className="model-stat reveal">
                <span>PREDICTIONS</span>

                <strong>
                  {totalPredictions}
                </strong>

                <small>
                  Recorded predictions
                </small>
              </div>
            </div>
          </div>

          {/* LIVE PREDICTION INTELLIGENCE */}

          <div className={`prediction-intelligence reveal ${
            newPrediction ? "intelligence-updated" : ""
          }`}>
            <div className="intelligence-header">
              <div>
                <span>03 · PREDICTION INTELLIGENCE</span>
                <h3>WHY THE MARKET SIGNAL LOOKS THIS WAY.</h3>
                <p>
                  Live technical and market conditions calculated from
                  the latest available NIFTY 50 data.
                </p>
              </div>

              <div className={`decision-chip ${
                isUp ? "decision-up" : "decision-down"
              }`}>
                <small>MODEL DECISION</small>
                <strong>
                  {isUp ? "▲ UP" : "▼ DOWN"}
                </strong>
                <span>
                  {confidence.toFixed(2)}% CONFIDENCE
                </span>
              </div>
            </div>

            <div className="intelligence-body">
              <div className="signal-stack">
                <div className="signal-item">
                  <span>TREND</span>
                  <strong className={
                    intelligence.trend === "BULLISH"
                      ? "green-text"
                      : intelligence.trend === "BEARISH"
                      ? "red-text"
                      : ""
                  }>
                    {intelligence.trend}
                  </strong>
                  <small>
                    Price vs moving-average structure
                  </small>
                </div>

                <div className="signal-item">
                  <span>MOMENTUM</span>
                  <strong className={
                    intelligence.momentum > 0
                      ? "green-text"
                      : intelligence.momentum < 0
                      ? "red-text"
                      : ""
                  }>
                    {Number.isFinite(intelligence.momentum)
                      ? `${intelligence.momentum >= 0 ? "+" : ""}${intelligence.momentum.toFixed(2)}%`
                      : "—"}
                  </strong>
                  <small>
                    Five-session price movement
                  </small>
                </div>

                <div className="signal-item">
                  <span>RSI 14</span>
                  <strong>
                    {intelligence.rsi == null
                      ? "—"
                      : intelligence.rsi.toFixed(1)}
                  </strong>
                  <small>
                    Momentum oscillator
                  </small>
                </div>

                <div className="signal-item">
                  <span>VOLATILITY</span>
                  <strong>
                    {intelligence.volatility == null
                      ? "—"
                      : `${intelligence.volatility.toFixed(1)}%`}
                  </strong>
                  <small>
                    Annualized rolling volatility
                  </small>
                </div>
              </div>

              <div className="signal-bars">
                <SignalBar
                  label="RSI POSITION"
                  value={rsiBar}
                  tone={rsiTone}
                />

                <SignalBar
                  label="MOMENTUM"
                  value={momentumBar}
                  tone={
                    intelligence.momentum >= 0
                      ? "green"
                      : "red"
                  }
                />

                <SignalBar
                  label="VOLATILITY"
                  value={volatilityBar}
                  tone="gold"
                />

                <SignalBar
                  label="VOLUME ACTIVITY"
                  value={volumeBar}
                  tone="green"
                />
              </div>

              <div className="probability-orbit">
                <div
                  className="probability-orbit-ring"
                  style={{
                    "--prediction-probability": `${
                      isUp
                        ? probabilityUp
                        : probabilityDown
                    }%`,
                  }}
                >
                  <div>
                    <strong>
                      {(isUp
                        ? probabilityUp
                        : probabilityDown
                      ).toFixed(1)}%
                    </strong>
                    <span>
                      {isUp ? "UP" : "DOWN"}
                    </span>
                  </div>
                </div>

                <small>
                  Directional probability
                </small>
              </div>
            </div>

            <div className="intelligence-footer">
              <span>
                LIVE MARKET FEATURES
              </span>

              <div>
                <b>RSI</b>
                <b>5D MOMENTUM</b>
                <b>VOLATILITY</b>
                <b>VOLUME</b>
                <b>MA STRUCTURE</b>
              </div>
            </div>
          </div>

          <div className="feature-grid">
            {[
              [
                "TECHNICAL",
                "RSI · MACD · Moving Averages · ATR",
              ],
              [
                "MOMENTUM",
                "Returns · Trend · Price Momentum",
              ],
              [
                "EXTERNAL",
                "VIX · S&P 500 · Gold · Crude",
              ],
              [
                "VOLATILITY",
                "ATR · Rolling Volatility · Range",
              ],
            ].map(([title, text]) => (
              <button
                className="feature-card reveal"
                key={title}
                onClick={() =>
                  showMessage(text)
                }
              >
                <span>{title}</span>
                <p>{text}</p>
              </button>
            ))}
          </div>
        </section>

        {/* TECHNICAL TERMINAL */}
        <section id="technical" className="section dark-section technical-terminal-section">
          <SectionTitle
            eyebrow="03 · TECHNICAL TERMINAL"
            title="READ THE MARKET STRUCTURE."
            description="A compact technical matrix derived from the same market history used by the dashboard intelligence layer."
          />

          <div className="technical-terminal-grid">
            <article className="terminal-chart-panel reveal">
              <div className="panel-header">
                <div>
                  <span>PRICE ACTION</span>
                  <small>NIFTY 50 · SELECTED WINDOW</small>
                </div>
                <div className="terminal-price-tag">
                  <b>{nifty}</b>
                  <span className={directionClass(dayChangePct)}>{dayChangePct >= 0 ? "+" : ""}{dayChangePct.toFixed(2)}%</span>
                </div>
              </div>
              <div className="terminal-chart">
                <Sparkline prices={prices} large />
              </div>
              <div className="terminal-chart-footer">
                <span>{chartRows[0]?.Date || chartRows[0]?.date || "—"}</span>
                <span>{chartRows.at(-1)?.Date || chartRows.at(-1)?.date || "—"}</span>
              </div>
            </article>

            <article className="indicator-stack reveal">
              {[
                ["RSI 14", Number.isFinite(productionRSI) ? productionRSI : advanced.rsi14, 100, productionRSI >= 70 ? "red" : productionRSI <= 30 ? "green" : "gold"],
                ["BB POSITION", advanced.bbPosition, 100, "gold"],
                ["STOCHASTIC", advanced.stochastic, 100, advanced.stochastic > 80 ? "red" : advanced.stochastic < 20 ? "green" : "gold"],
                ["VOLUME ACTIVITY", advanced.volumeRatio != null ? Math.min(100, advanced.volumeRatio * 50) : null, 100, "green"],
              ].map(([label, value, max, tone]) => (
                <div className="indicator-row" key={label}>
                  <div><span>{label}</span><b>{Number.isFinite(Number(value)) ? `${Number(value).toFixed(1)}${label === "VOLUME ACTIVITY" ? "x" : ""}` : "—"}</b></div>
                  <div className="indicator-track"><i className={tone} style={{ width: `${Number.isFinite(Number(value)) ? Math.max(0, Math.min(max, Number(value))) : 0}%` }} /></div>
                </div>
              ))}
            </article>
          </div>
        </section>

        {/* TRACKING */}

        <section className="section tracking-section">
          <SectionTitle
            eyebrow="03 · LIVE TRACKING"
            title="WE KEEP THE RECEIPTS."
            description="Every prediction becomes part of the historical record."
          />

          <div className="tracking-grid">
            {[
              [
                totalPredictions,
                "TOTAL PREDICTIONS",
              ],
              [verified, "VERIFIED"],
              [correct, "CORRECT"],
              [
                accuracy == null
                  ? "—"
                  : `${accuracy.toFixed(1)}%`,
                "LIVE ACCURACY",
              ],
            ].map(([value, label]) => (
              <div
                className="tracking-number reveal"
                key={label}
              >
                <strong>{value}</strong>
                <span>{label}</span>
              </div>
            ))}
          </div>

          <div className="tracking-actions">
            <button
              className="gold-button"
              onClick={runPipeline}
              disabled={pipelineRunning}
            >
              {pipelineRunning
                ? "PIPELINE RUNNING..."
                : "RUN DAILY PIPELINE"}
            </button>

            <button
              className="outline-button"
              onClick={() =>
                download(
                  "download/history"
                )
              }
            >
              DOWNLOAD HISTORY
            </button>
          </div>
        </section>

        {/* BACKTEST */}

        <section className="section backtest-section">
          <SectionTitle
            eyebrow="04 · QUANTITATIVE PERFORMANCE"
            title="TESTING THE THEORY."
            description="Historical performance is presented as an experimental research result."
          />

          <div className="backtest-grid">
            {[
              [
                "+13.94%",
                "TECHNICAL MODEL",
                70,
              ],
              [
                "+24.63%",
                "EXTERNAL FEATURES",
                82,
              ],
              [
                "+48.81%",
                "BUY & HOLD",
                100,
              ],
              [
                "0.608",
                "EXTERNAL SHARPE",
                61,
              ],
            ].map(
              ([value, title, width]) => (
                <button
                  className="backtest-card reveal"
                  key={title}
                  onClick={() =>
                    showMessage(
                      `${title}: historical research metric`
                    )
                  }
                >
                  <span>{title}</span>

                  <strong>{value}</strong>

                  <p>
                    Historical research metric
                  </p>

                  <div className="performance-line">
                    <i
                      style={{
                        width: `${width}%`,
                      }}
                    />
                  </div>
                </button>
              )
            )}
          </div>

          <div className="research-warning reveal">
            <span>!</span>

            <div>
              <b>
                RESEARCH RESULT, NOT FINANCIAL ADVICE
              </b>

              <p>
                Historical backtests do not guarantee
                future market performance.
              </p>
            </div>
          </div>
        </section>

        {/* RESEARCH */}

        <section
          id="research"
          className="section"
        >
          <SectionTitle
            eyebrow="05 · RESEARCH"
            title="THE SCIENCE BEHIND THE SIGNAL."
            description="Prediction is treated as an experimental research problem."
          />

          <div className="research-grid">
            {[
              [
                "01",
                "DATA",
                "Historical NIFTY 50 market data.",
              ],
              [
                "02",
                "FEATURE ENGINEERING",
                "Returns, momentum and volatility.",
              ],
              [
                "03",
                "MACHINE LEARNING",
                "CatBoost production directional classification.",
              ],
              [
                "04",
                "VALIDATION",
                "Walk-forward out-of-sample testing.",
              ],
              [
                "05",
                "BACKTESTING",
                "Historical strategy evaluation.",
              ],
              [
                "06",
                "VERIFICATION",
                "Future outcomes compared with signals.",
              ],
            ].map(
              ([number, title, text]) => (
                <button
                  className="research-card reveal"
                  key={number}
                  onClick={() =>
                    showMessage(
                      `${title}: ${text}`
                    )
                  }
                >
                  <span>{number}</span>

                  <h3>{title}</h3>

                  <p>{text}</p>

                  <b>EXPLORE →</b>
                </button>
              )
            )}
          </div>

          <div className="methodology reveal">
            <div>
              <span>
                VALIDATION FRAMEWORK
              </span>

              <h3>
                WALK-FORWARD
                <br />
                VALIDATION
              </h3>
            </div>

            <div className="method-flow">
              <div>
                TRAIN
                <small>
                  Historical Window
                </small>
              </div>

              <i>→</i>

              <div>
                TEST
                <small>
                  Future Window
                </small>
              </div>

              <i>→</i>

              <div>
                VERIFY
                <small>
                  Actual Outcome
                </small>
              </div>
            </div>
          </div>
        </section>

        {/* HISTORY */}

        <section
          id="history"
          className="section"
        >
          <SectionTitle
            eyebrow="06 · PREDICTION HISTORY"
            title="THE RECORD."
            description="Predictions can be reviewed against future market outcomes."
          />

          <div className="history-panel reveal">
            {history.length === 0 ? (
              <div className="empty-history">
                NO PREDICTION HISTORY AVAILABLE.
              </div>
            ) : (
              <div className="history-list">
                {history
                  .slice(-10)
                  .reverse()
                  .map((row, index) => {
                    const direction =
                      String(
                        row.Prediction ||
                          row.prediction ||
                          ""
                      ).toUpperCase();

                    const confidenceValue =
                      row.Confidence ??
                      row.confidence;

                    return (
                      <div
                        className="history-row"
                        key={index}
                      >
                        <span>
                          {row.Date ||
                            row.date ||
                            "—"}
                        </span>

                        <strong
                          className={
                            direction === "UP"
                              ? "green-text"
                              : "red-text"
                          }
                        >
                          {direction || "—"}
                        </strong>

                        <span>
                          CONFIDENCE
                        </span>

                        <b>
                          {confidenceValue != null
                            ? `${(
                                asPercent(confidenceValue)
                              ).toFixed(2)}%`
                            : "—"}
                        </b>

                        <span className="history-status">
                          RECORDED
                        </span>
                      </div>
                    );
                  })}
              </div>
            )}

            <button
              className="text-link"
              onClick={() =>
                download(
                  "download/history"
                )
              }
            >
              DOWNLOAD COMPLETE HISTORY →
            </button>
          </div>
        </section>

        {/* ABOUT */}

        <section
          id="about"
          className="section about-section"
        >
          <div className="about-grid">
            <div className="about-copy reveal">
              <span className="eyebrow">
                07 · MURTAZA LABS
              </span>

              <h2>
                DISCIPLINE
                <br />
                OVER NOISE.
              </h2>

              <p>
                Murtaza Labs is a personal
                quantitative research environment
                exploring machine learning, data and
                systematic experimentation in
                financial markets.
              </p>

              <p>
                NIFTY 50 Intelligence focuses on
                directional prediction, validation
                and reproducible research.
              </p>

              <div className="about-signature">
                MOHAMMED MURTAZA TAJAMMUL

                <small>
                  CSE · OSMANIA UNIVERSITY
                </small>
              </div>
            </div>

            <IntelligenceVisual />
          </div>
        </section>

        {/* FINAL */}

        <section className="final-section">
          <span>THE NEXT PREDICTION</span>

          <h2>
            KEEP
            <strong>BUILDING.</strong>
          </h2>

          <p>
            Data changes. Markets change. The
            system keeps learning.
          </p>

          <button
            className="gold-button"
            onClick={() =>
              scrollTo("market")
            }
          >
            RETURN TO MARKET
          </button>
        </section>
      </main>

      {/* FOOTER */}

      <footer>
        <div>
          <b>MURTAZA LABS</b>
          <span>NIFTY 50 INTELLIGENCE</span>
        </div>

        <div className="footer-center">
          Built & Researched by

          <strong>
            Mohammed Murtaza Tajammul
          </strong>

          <small>
            CSE · Muffakham Jah College of
            Engineering & Technology ·
            Osmania University
          </small>
        </div>

        <div className="footer-quote">
          DISCIPLINE
          <br />
          DATA
          <br />
          RESEARCH
        </div>
      </footer>

      {showTop && (
        <button
          className="top-button"
          onClick={() =>
            window.scrollTo({
              top: 0,
              behavior: "smooth",
            })
          }
        >
          ↑
        </button>
      )}

      <button
        className="refresh-button"
        onClick={() => loadData(true)}
        disabled={refreshing}
      >
        {refreshing ? "…" : "↻"}
      </button>

      {message && (
        <button
          className="toast"
          onClick={() => setMessage("")}
        >
          <i />
          {message}
          <b>×</b>
        </button>
      )}
    </div>
  );
}