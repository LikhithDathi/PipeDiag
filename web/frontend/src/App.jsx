import { useCallback, useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./App.css";

const API = "/api";

const NAV = [
  ["overview", "Overview"],
  ["findings", "Findings"],
  ["evidence", "Evidence"],
  ["runtime", "Runtime"],
  ["evaluations", "Evaluations"],
  ["report", "Report"],
  ["history", "Run History"],
];

const pct = (value) => {
  const n = Number(value);
  return Number.isFinite(n) ? `${(n * 100).toFixed(1)}%` : "—";
};

const titleCase = (value) =>
  String(value ?? "Unknown")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());

const duration = (value) => {
  const n = Number(value);
  return Number.isFinite(n) ? `${n.toFixed(2)} ms` : "—";
};

const pretty = (value) => {
  if (value === null || value === undefined) return "—";
  return typeof value === "object" ? JSON.stringify(value) : String(value);
};

function Icon({ name }) {
  const paths = {
    overview: (
      <>
        <rect x="3" y="3" width="7" height="7" rx="1" />
        <rect x="14" y="3" width="7" height="7" rx="1" />
        <rect x="3" y="14" width="7" height="7" rx="1" />
        <rect x="14" y="14" width="7" height="7" rx="1" />
      </>
    ),
    findings: (
      <>
        <path d="M4 19V5" />
        <path d="M4 19h17" />
        <path d="m7 15 4-4 3 2 5-7" />
      </>
    ),
    evidence: (
      <>
        <path d="M5 3h14v18H5z" />
        <path d="M8 8h8M8 12h8M8 16h5" />
      </>
    ),
    runtime: (
      <>
        <path d="M3 12h4l3-8 4 16 3-8h4" />
      </>
    ),
    evaluations: (
      <>
        <path d="M4 19V5M4 19h17" />
        <path d="M8 16v-4M13 16V7M18 16v-7" />
      </>
    ),
    report: (
      <>
        <path d="M6 3h9l4 4v14H6z" />
        <path d="M15 3v5h5M9 12h7M9 16h7" />
      </>
    ),
    history: (
      <>
        <path d="M3 12a9 9 0 1 0 2.6-6.4L3 8" />
        <path d="M3 3v5h5" />
        <path d="M12 7v5l3 2" />
      </>
    ),
  };

  return (
    <svg
      viewBox="0 0 24 24"
      aria-hidden="true"
      className="nav-icon"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      {paths[name]}
    </svg>
  );
}

function App() {
  const [health, setHealth] = useState("checking");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [activeTab, setActiveTab] = useState("overview");
  const [runId, setRunId] = useState("");
  const [showAllHypotheses, setShowAllHypotheses] = useState(false);

  // Run history state
  const [history, setHistory] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState("");
  const [deletingRunId, setDeletingRunId] = useState("");

  const refreshHistory = useCallback(async () => {
    setHistoryLoading(true);
    setHistoryError("");

    try {
      const response = await fetch(`${API}/runs`);

      if (!response.ok) {
        throw new Error(`History request failed (${response.status}).`);
      }

      const data = await response.json();

      if (!Array.isArray(data)) {
        throw new Error("Invalid run history response.");
      }

      setHistory(data);
    } catch (err) {
      setHistoryError(err.message || "Unable to load run history.");
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  const load = useCallback(
    async (url, options = {}, label = "Request") => {
      setLoading(true);
      setError("");

      try {
        const response = await fetch(url, options);

        if (!response.ok) {
          const detail =
            response.status === 404
              ? "Run not found."
              : `${label} failed (${response.status}).`;

          throw new Error(detail);
        }

        const data = await response.json();

        setResult(data);
        setRunId(data.diagnostic?.metadata?.run_id ?? runId);
        setActiveTab("overview");
        setShowAllHypotheses(false);

        // Refresh saved history after loading or creating a run.
        refreshHistory();
      } catch (err) {
        setError(
          `${err.message} Check that the PipeDiag API is running.`
        );
      } finally {
        setLoading(false);
      }
    },
    [runId, refreshHistory]
  );

  const handleDeleteRun = async (id) => {
    if (!window.confirm(`Delete diagnostic run ${id}?`)) {
      return;
    }

    setDeletingRunId(id);
    setHistoryError("");

    try {
      const response = await fetch(
        `${API}/runs/${encodeURIComponent(id)}`,
        { method: "DELETE" }
      );

      if (!response.ok) {
        const detail =
          response.status === 404
            ? "Run not found."
            : `Delete failed (${response.status}).`;

        throw new Error(detail);
      }

      if (result?.diagnostic?.metadata?.run_id === id) {
        setResult(null);
        setRunId("");
        setActiveTab("history");
      }

      await refreshHistory();
    } catch (err) {
      setHistoryError(err.message || "Unable to delete run.");
    } finally {
      setDeletingRunId("");
    }
  };

  useEffect(() => {
    fetch(`${API}/health`)
      .then((r) => {
        if (!r.ok) throw new Error();
        setHealth("online");
      })
      .catch(() => setHealth("offline"));
  }, []);

  useEffect(() => {
    let cancelled = false;

    const loadInitialHistory = async () => {
      // Defer execution until after the effect has completed.
      await Promise.resolve();

      if (!cancelled) {
        refreshHistory();
      }
    };

    loadInitialHistory();

    return () => {
      cancelled = true;
    };
  }, [refreshHistory]);

  const diagnostic = result?.diagnostic ?? {};
  const explanation = result?.explanation ?? {};
  const runtime = result?.runtime_summary ?? {};
  const events = result?.runtime_events ?? [];
  const evaluations = result?.evaluations ?? [];
  const evidence = result?.evidence ?? [];
  const threshold = Number(explanation.threshold ?? 0.2);

  // Detect whether this is a multi-fault diagnostic run.
  const multiFault = diagnostic.multi_fault ?? null;
  const isMultiFault = Boolean(multiFault);

  // Select the correct posterior distribution.
  // Multi-fault: normalized multi-fault support.
  // Single-fault: original categorical posterior.
  const posterior = isMultiFault
    ? multiFault.posterior ?? {}
    : diagnostic.posterior ?? {};

  // Multi-fault detection is independent of normalized support.
  const detectedHypotheses =
    multiFault?.detected_hypotheses ?? [];

  const evidenceSupportedHypotheses =
    multiFault?.evidence_supported_hypotheses ?? [];

  const hypotheses = useMemo(() => {
    return Object.entries(posterior)
      .map(([id, value]) => {
        const detail = (explanation.hypotheses ?? []).find(
          (h) => h.hypothesis_id === id
        );

        return {
          id,
          title: detail?.title ?? id,

          // This is the value displayed in the dashboard.
          value: Number(value),

          // Detection must NOT use the normalized percentage.
          detected: isMultiFault
            ? detectedHypotheses.includes(id)
            : Number(value) >= threshold,

          evidence: detail?.evidence ?? [],

          evidenceSupported: isMultiFault
            ? evidenceSupportedHypotheses.includes(id)
            : detail?.evidence_supported ?? false,

          points: detail?.explanation_points ?? [],
          recommendation: detail?.recommendation ?? "",
          detail: Boolean(detail),
        };
      })
      .sort((a, b) => b.value - a.value);
  }, [
    posterior,
    explanation.hypotheses,
    isMultiFault,
    detectedHypotheses,
    evidenceSupportedHypotheses,
    threshold,
  ]);

  const qualifying = hypotheses.filter(
    (hypothesis) => hypothesis.detected
  );

  const supportedFindings = qualifying.filter(
    (h) => h.evidenceSupported
  );

  const priorOnlyHypotheses = qualifying.filter(
    (h) => !h.evidenceSupported
  );

  const inconclusive = isMultiFault
    ? qualifying.length === 0
    : Boolean(explanation.inconclusive);

  const failed = Number(
    runtime.failed_events ??
      events.filter((e) =>
        ["failed", "error"].includes(String(e.status).toLowerCase())
      ).length
  );

  const successful = Number(
    runtime.successful_events ??
      events.filter(
        (e) => String(e.status).toLowerCase() === "success"
      ).length
  );

  const total = Number(runtime.total_events ?? events.length);
  const executionRate =
    total > 0 ? (successful / total) * 100 : null;

  const observedGroups = evidence.filter(
    (e) => String(e.status).toLowerCase() === "observed"
  ).length;

  const runType =
    result?.run_type === "demo"
      ? "Demo run"
      : result?.run_type === "example"
        ? "Example run"
        : "Diagnostic run";

  const submitRun = (event) => {
    event.preventDefault();

    if (runId.trim()) {
      load(
        `${API}/runs/${encodeURIComponent(runId.trim())}`,
        {},
        "Run lookup"
      );
    }
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a
          className="brand"
          href="#top"
          onClick={() => setActiveTab("overview")}
        >
          <span className="brand-mark">P</span>
          <span>
            <strong>PipeDiag</strong>
            <small>PIPELINE INTELLIGENCE</small>
          </span>
        </a>

        <div className="side-caption">WORKSPACE</div>

        <nav className="side-nav">
          {NAV.map(([id, label]) => (
            <button
              key={id}
              className={`nav-item ${
                activeTab === id ? "active" : ""
              }`}
              onClick={() => setActiveTab(id)}
            >
              <Icon name={id} />
              <span>{label}</span>
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="connection">
            <span className={`connection-dot ${health}`} />
            <span>
              <b>API connection</b>
              <small>
                {health === "online"
                  ? "Backend reachable"
                  : health === "offline"
                    ? "Backend unavailable"
                    : "Checking connection"}
              </small>
            </span>
          </div>

          <div className="side-foot">
            Explainable ML pipeline diagnostics
          </div>
        </div>
      </aside>

      <main className="main" id="top">
        <header className="topbar">
          <div className="page-heading">
            <div className="eyebrow">
              PIPELINE INTELLIGENCE <span>/</span>{" "}
              {titleCase(activeTab)}
            </div>

            <h1>
              {activeTab === "overview"
                ? "Diagnostic overview"
                : NAV.find(([id]) => id === activeTab)?.[1]}
            </h1>

            <p>
              Evidence-led insight into machine learning pipeline
              behaviour.
            </p>
          </div>

          <div className="actions">
            <form className="run-search" onSubmit={submitRun}>
              <input
                aria-label="Run ID"
                placeholder="Enter run ID"
                value={runId}
                onChange={(e) => setRunId(e.target.value)}
              />
              <button type="submit" disabled={loading}>
                Open run
              </button>
            </form>
          </div>
        </header>

        {error && (
          <div className="error-banner">
            <b>Unable to load diagnostics</b>
            <span>{error}</span>
          </div>
        )}

        {/* Run History view */}
        {activeTab === "history" && (
          <RunHistory
            history={history}
            loading={historyLoading}
            error={historyError}
            currentRunId={result?.diagnostic?.metadata?.run_id}
            deletingRunId={deletingRunId}
            onRefresh={refreshHistory}
            onOpen={(id) =>
              load(
                `${API}/runs/${encodeURIComponent(id)}`,
                {},
                "Run lookup"
              )
            }
            onDelete={handleDeleteRun}
          />
        )}

        {/* Empty workspace */}
        {!result && !error && activeTab !== "history" && (
          <section className="empty-workspace">
            <div className="empty-mark">P</div>
            <h2>Your diagnostic workspace</h2>
            <p>
              <p>
                Open a saved diagnostic run using its run ID or select one
                from Run History to inspect execution, evidence and findings.
              </p>
            </p>
          </section>
        )}

        {/* Existing diagnostic dashboard */}
        {result && activeTab !== "history" && (
          <>
            <div className="run-meta">
              <span>{runType}</span>
              <i />
              <span>
                {diagnostic.metadata?.run_id ?? "Run ID unavailable"}
              </span>
              <i />
              <span>{total} captured events</span>
              <span className="run-meta-spacer" />

              <span
                className={`status-label ${
                  inconclusive ? "attention" : "neutral"
                }`}
              >
                {inconclusive
                  ? "INCONCLUSIVE"
                  : "ASSESSMENT COMPLETE"}
              </span>
            </div>

            {activeTab === "overview" && (
              <Overview
                result={result}
                hypotheses={hypotheses}
                qualifying={qualifying}
                supportedFindings={supportedFindings}
                priorOnlyHypotheses={priorOnlyHypotheses}
                threshold={threshold}
                isMultiFault={isMultiFault}
                inconclusive={inconclusive}
                events={events}
                evaluations={evaluations}
                evidence={evidence}
                runtime={runtime}
                successful={successful}
                failed={failed}
                total={total}
                executionRate={executionRate}
                observedGroups={observedGroups}
                showAll={showAllHypotheses}
                setShowAll={setShowAllHypotheses}
                go={setActiveTab}
              />
            )}

            {activeTab === "findings" && (
              <Findings
                hypotheses={hypotheses}
                qualifying={qualifying}
                threshold={threshold}
                isMultiFault={isMultiFault}
                explanation={explanation}
              />
            )}

            {activeTab === "evidence" && (
              <EvidenceView
                evidence={evidence}
                hypotheses={hypotheses}
              />
            )}

            {activeTab === "runtime" && (
              <RuntimeView events={events} runtime={runtime} />
            )}

            {activeTab === "evaluations" && (
              <EvaluationView evaluations={evaluations} />
            )}

            {activeTab === "report" && (
              <ReportView report={result.report} />
            )}

            <footer className="footer">
              <span>PipeDiag · Explainable ML diagnostics</span>
              <span>
                Displayed values are sourced from the diagnostic API
                response.
              </span>
            </footer>
          </>
        )}
      </main>
    </div>
  );
}

/* ============================================================
   RUN HISTORY
   ============================================================ */

function RunHistory({
  history,
  loading,
  error,
  currentRunId,
  deletingRunId,
  onRefresh,
  onOpen,
  onDelete,
}) {
  const formatDate = (value) => {
    if (!value) return "Unknown date";

    const date = new Date(value);

    return Number.isNaN(date.getTime())
      ? "Unknown date"
      : date.toLocaleString();
  };

  return (
    <section className="surface page-surface history-page">
      <div className="page-section-heading">
        <div>
          <div className="section-kicker">SAVED DIAGNOSTIC RUNS</div>
          <h2>Run history</h2>
          <p>
            Browse, reopen and manage previously saved diagnostic
            runs.
          </p>
        </div>

        <button
          className="button button-quiet"
          onClick={onRefresh}
          disabled={loading}
        >
          {loading ? "Refreshing…" : "Refresh history"}
        </button>
      </div>

      {error && (
        <div className="error-banner">
          <b>History unavailable</b>
          <span>{error}</span>
        </div>
      )}

      {loading && history.length === 0 && (
        <Empty>Loading saved diagnostic runs…</Empty>
      )}

      {!loading && history.length === 0 && !error && (
        <div className="empty-large">
          <div className="empty-symbol">◷</div>
          <h3>No saved runs yet</h3>
          <p>
            Execute and save a diagnostic run to see it appear here.
          </p>
        </div>
      )}

      {history.length > 0 && (
        <>
          <div className="history-summary">
            <span>{history.length} saved runs</span>
            <span>Newest first</span>
          </div>

          <div className="history-list">
            {history.map((run) => (
              <article className="history-item" key={run.run_id}>
                <div className="history-item-main">
                  <div className="history-item-heading">
                    <b>{run.run_id}</b>

                    {currentRunId === run.run_id && (
                      <span className="history-current">
                        CURRENT
                      </span>
                    )}
                  </div>

                  <span className="history-date">
                    {formatDate(run.saved_at)}
                  </span>
                </div>

                <div className="history-metrics">
                  <div>
                    <span>Type</span>
                    <b>{titleCase(run.run_type)}</b>
                  </div>

                  <div>
                    <span>Uncertainty</span>
                    <b>{pct(run.uncertainty)}</b>
                  </div>

                  <div>
                    <span>Events</span>
                    <b>{run.total_events ?? 0}</b>
                  </div>
                </div>

                <div className="history-actions">
                  <button
                    className="button button-primary"
                    onClick={() => onOpen(run.run_id)}
                    disabled={deletingRunId === run.run_id}
                  >
                    Open run
                  </button>

                  <button
                    className="button button-quiet history-delete"
                    onClick={() => onDelete(run.run_id)}
                    disabled={deletingRunId === run.run_id}
                  >
                    {deletingRunId === run.run_id
                      ? "Deleting…"
                      : "Delete"}
                  </button>
                </div>
              </article>
            ))}
          </div>
        </>
      )}
    </section>
  );
}

/* ============================================================
   OVERVIEW
   ============================================================ */

function Overview(props) {
  const {
    result,
    hypotheses,
    qualifying,
    supportedFindings,
    priorOnlyHypotheses,
    threshold,
    isMultiFault,
    inconclusive,
    events,
    evaluations,
    evidence,
    runtime,
    successful,
    failed,
    total,
    executionRate,
    observedGroups,
    showAll,
    setShowAll,
    go,
  } = props;

  const top = supportedFindings;
  const leading = hypotheses[0];

  return (
    <div className="overview-layout">
      <section className="overview-top">
        <article className="assessment-panel">
          <div className="section-kicker">
            DIAGNOSTIC ASSESSMENT
            <span
              className={`assessment-state ${
                inconclusive ? "amber" : "sage"
              }`}
            >
              {inconclusive ? "INCONCLUSIVE" : "REVIEW FINDINGS"}
            </span>
          </div>

          <h2>
            {inconclusive
              ? "The available evidence is not sufficient for a conclusive diagnosis."
              : "Diagnostic assessment is available for review."}
          </h2>

          <p>
            {result.explanation?.overall_assessment ||
              "Review the evidence and hypothesis distribution to understand what this run establishes."}
          </p>

          <div className="assessment-foot">
            <span className="tiny-dot amber-dot" />
            <span>
              {qualifying.length
                ? `${qualifying.length} hypothesis${
                    qualifying.length === 1 ? "" : "es"
                  } meet the configured threshold.`
                : "No hypothesis meets the configured threshold."}
            </span>
          </div>
        </article>

        <article className="execution-panel">
          <div className="section-kicker">
            EXECUTION HEALTH
            <span className="subtle-label">{total} EVENTS</span>
          </div>

          <div className="health-visual">
            <div className="health-number">
              {executionRate === null
                ? "—"
                : `${executionRate.toFixed(0)}%`}
              <span> successful</span>
            </div>

            <div
              className="health-donut"
              style={{
                "--success-angle": `${
                  total ? (successful / total) * 360 : 0
                }deg`,
                "--failure-angle": `${
                  total ? ((successful + failed) / total) * 360 : 0
                }deg`,
              }}
            >
              <div>
                <b>
                  {successful}/{total}
                </b>
                <small>events</small>
              </div>
            </div>
          </div>

          <div className="health-track">
            <span
              className="health-success"
              style={{
                width: `${total ? (successful / total) * 100 : 0}%`,
              }}
            />
            <span
              className="health-failure"
              style={{
                width: `${total ? (failed / total) * 100 : 0}%`,
              }}
            />
          </div>

          <div className="health-legend">
            <span>
              <i className="legend-sage" /> {successful} successful
            </span>
            <span>
              <i className="legend-red" /> {failed} failed
            </span>
          </div>

          <div className="execution-foot">
            <span>Total duration</span>
            <b>{duration(runtime.total_duration_ms)}</b>
          </div>
        </article>
      </section>

      <section className="surface signal-panel">
        <div className="panel-title">
          <div>
            <div className="section-kicker">POSTERIOR SIGNAL</div>
            <h3>Hypothesis probability landscape</h3>
            <p>
              Relative posterior support across the diagnostic
              hypotheses. The vertical marker is the configured
              threshold.
            </p>
          </div>

          <button
            className="text-action"
            onClick={() => go("findings")}
          >
            Explore distribution <span>↗</span>
          </button>
        </div>

        {hypotheses.length ? (
          <div className="signal-chart">
            <div className="signal-axis">
              <span>0%</span>
              <span>25%</span>
              <span>50%</span>
              <span>75%</span>
              <span>100%</span>
            </div>

            <div className="signal-rows">
              {hypotheses.slice(0, 9).map((h) => (
                <div className="signal-row" key={h.id}>
                  <span className="signal-label">{h.id}</span>

                  <div className="signal-track">
                    <span
                      className={`signal-fill ${
                        h.detected
                          ? "signal-qualified"
                          : "signal-muted"
                      }`}
                      style={{
                        width: `${Math.max(
                          0,
                          Math.min(100, h.value * 100)
                        )}%`,
                      }}
                    />

                    {!isMultiFault && (
                      <i
                        className="threshold-marker"
                        style={{
                          left: `${Math.max(
                            0,
                            Math.min(100, threshold * 100)
                          )}%`,
                        }}
                      />
                    )}
                  </div>

                  <b
                    className={
                      h.detected
                        ? "signal-value qualified-value"
                        : "signal-value"
                    }
                  >
                    {pct(h.value)}
                  </b>
                </div>
              ))}
            </div>

            <div className="signal-legend">
              <span>
                <i className="legend-green" />
                {isMultiFault ? "Detected" : "At / above threshold"}
              </span>

              <span>
                <i className="legend-grey" />
                {isMultiFault ? "Not detected" : "Below threshold"}
              </span>

              {!isMultiFault && (
                <span>
                  <i className="legend-amber" /> Threshold · {pct(threshold)}
                </span>
              )}
            </div>
          </div>
        ) : (
          <Empty>No posterior distribution was returned.</Empty>
        )}
      </section>

      <section className="overview-grid">
        <article className="surface trace-panel">
          <div className="panel-title">
            <div>
              <div className="section-kicker">PIPELINE TRACE</div>
              <h3>Execution sequence</h3>
            </div>

            <button
              className="text-action"
              onClick={() => go("runtime")}
            >
              View runtime <span>↗</span>
            </button>
          </div>

          {events.length ? (
            <div className="trace">
              {events.map((event, i) => (
                <div
                  className="trace-step"
                  key={event.event_id ?? i}
                >
                  <div
                    className={`trace-node ${
                      String(event.status).toLowerCase() === "success"
                        ? "done"
                        : "other"
                    }`}
                  >
                    {i + 1}
                  </div>

                  <div className="trace-content">
                    <b>{event.operation ?? "Operation"}</b>
                    <span>
                      {event.component ??
                        event.step_name ??
                        "Pipeline component"}
                    </span>
                  </div>

                  <div className="trace-duration">
                    {duration(event.duration_ms)}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <Empty>
              No runtime events were captured for this run.
            </Empty>
          )}
        </article>

        <article className="surface coverage-panel">
          <div className="panel-title">
            <div>
              <div className="section-kicker">EVIDENCE ACQUISITION</div>
              <h3>Coverage overview</h3>
            </div>

            <button
              className="text-action"
              onClick={() => go("evidence")}
            >
              Inspect <span>↗</span>
            </button>
          </div>

          <div className="coverage-count">
            <strong>{observedGroups}</strong>
            <span>of 8 groups observed</span>
          </div>

          <div className="coverage-grid">
            {Array.from({ length: 8 }, (_, i) => {
              const group = evidence.find((item) =>
                String(item.evidence_type)
                  .toUpperCase()
                  .endsWith(String(i + 1))
              );

              const status = String(
                group?.status ?? "not returned"
              ).toLowerCase();

              return (
                <div
                  key={i}
                  className={`coverage-cell ${
                    status === "observed"
                      ? "observed"
                      : status === "insufficient"
                        ? "insufficient"
                        : ""
                  }`}
                  title={`G${i + 1}: ${status}`}
                >
                  <b>G{i + 1}</b>
                  <span>
                    {status === "observed"
                      ? "Observed"
                      : status === "insufficient"
                        ? "Insufficient"
                        : "Not returned"}
                  </span>
                </div>
              );
            })}
          </div>

          <p className="coverage-note">
            Not returned does not mean the condition was tested and
            passed.
          </p>
        </article>
      </section>

      <section className="surface hypothesis-panel">
        <div className="panel-title">
          <div>
            <div className="section-kicker">DIAGNOSTIC FINDINGS</div>
            <h3>Evidence-supported findings</h3>
            <p>
              Posterior support at or above {pct(threshold)}. These
              are not confirmed root causes.
            </p>
          </div>

          <button
            className="text-action"
            onClick={() => go("findings")}
          >
            All hypotheses <span>↗</span>
          </button>
        </div>

        {top.length ? (
          <div className="hypothesis-bars">
            {top.map((h) => (
              <div className="hypothesis-bar-row" key={h.id}>
                <div className="hypothesis-bar-label">
                  <b>{h.title}</b>
                  <span>{h.id}</span>
                </div>

                <div className="bar-rail">
                  <span
                    style={{
                      width: `${Math.max(
                        0,
                        Math.min(100, h.value * 100)
                      )}%`,
                    }}
                  />
                </div>

                <strong>{pct(h.value)}</strong>
              </div>
            ))}
          </div>
        ) : (
          <div className="no-threshold">
            <span className="tiny-dot amber-dot" />
            <div>
              <b>No hypothesis crossed the threshold</b>
              <p>
                {leading
                  ? `The highest posterior is ${pct(
                      leading.value
                    )} (${leading.id}), below the ${pct(
                      threshold
                    )} threshold.`
                  : "No posterior distribution was provided for this run."}
              </p>
            </div>
          </div>
        )}

        {priorOnlyHypotheses.length > 0 && (
          <div className="prior-only-section">
            <h4>Prior-only hypotheses</h4>
            <p>
              These hypotheses crossed the probability threshold but have
              no eligible supporting evidence in this run.
            </p>

            {priorOnlyHypotheses.map((h) => (
              <div className="hypothesis-bar-row" key={h.id}>
                <div className="hypothesis-bar-label">
                  <b>{h.title}</b>
                  <span>{h.id} · Prior-only</span>
                </div>

                <div className="bar-rail">
                  <span
                    className="prior-only-bar"
                    style={{ width: `${Math.min(100, h.value * 100)}%` }}
                  />
                </div>

                <strong>{pct(h.value)}</strong>
              </div>
            ))}
          </div>
        )}

        <button
          className="expand-hypotheses"
          onClick={() => setShowAll(!showAll)}
        >
          {showAll
            ? "Hide full distribution"
            : `View all ${hypotheses.length} hypotheses`}
          <span>{showAll ? "−" : "+"}</span>
        </button>

        {showAll && (
          <div className="all-hypotheses">
            {hypotheses.map((h) => (
              <div className="all-hypothesis-row" key={h.id}>
                <span>{h.id}</span>

                <div className="bar-rail">
                  <span
                    className={
                      h.value >= threshold ? "above" : "below"
                    }
                    style={{
                      width: `${Math.max(
                        0,
                        Math.min(100, h.value * 100)
                      )}%`,
                    }}
                  />
                </div>

                <b>{pct(h.value)}</b>

                <small>
                  {h.evidenceSupported
                    ? "Evidence-supported · above threshold"
                    : h.value >= threshold
                      ? "Prior-only · insufficient evidence"
                      : "Below threshold"}
                </small>
              </div>
            ))}
          </div>
        )}
      </section>

      <section className="overview-bottom">
        <article className="surface next-panel">
          <div className="section-kicker">INVESTIGATION</div>
          <h3>Where to look next</h3>

          {evaluations.length === 0 && (
            <div className="priority-item">
              <span className="priority-index">01</span>
              <div>
                <b>Record model evaluation metrics</b>
                <p>
                  No evaluation records were captured. Add an
                  appropriate evaluation step to assess model
                  performance.
                </p>
              </div>
              <button onClick={() => go("evaluations")}>↗</button>
            </div>
          )}

          <div className="priority-item">
            <span className="priority-index">
              {evaluations.length === 0 ? "02" : "01"}
            </span>
            <div>
              <b>Review acquired evidence</b>
              <p>
                Inspect measurements and acquisition status before
                drawing conclusions about unobserved conditions.
              </p>
            </div>
            <button onClick={() => go("evidence")}>↗</button>
          </div>
        </article>

        <article className="surface quick-panel">
          <div className="section-kicker">RUN SNAPSHOT</div>
          <h3>Captured information</h3>

          <div className="snapshot-row">
            <span>Runtime events</span>
            <b>{total}</b>
          </div>

          <div className="snapshot-row">
            <span>Evaluation records</span>
            <b>{evaluations.length}</b>
          </div>

          <div className="snapshot-row">
            <span>Evidence groups returned</span>
            <b>{evidence.length}</b>
          </div>

          <div className="snapshot-row">
            <span>Dataset references</span>
            <b>{runtime.dataset_reference_count ?? "—"}</b>
          </div>
        </article>
      </section>
    </div>
  );
}

/* ============================================================
   FINDINGS
   ============================================================ */

function Findings({
  hypotheses,
  qualifying,
  threshold,
  isMultiFault,
  explanation,
}) {
  const [showAll, setShowAll] = useState(false);
  const [selected, setSelected] = useState(null);

  const list = showAll ? hypotheses : qualifying;
  const current = hypotheses.find((h) => h.id === selected);

  return (
    <section className="surface page-surface">
      <div className="page-section-heading">
        <div>
          <div className="section-kicker">POSTERIOR DISTRIBUTION</div>
          <h2>Diagnostic hypotheses</h2>
          <p>
            {isMultiFault
              ? "Normalized support distribution. Detection is based on independent hypothesis probabilities."
              : `Threshold: ${pct(threshold)}. Posterior support does not establish causality.`}
          </p>
        </div>

        <button
          className="button button-quiet"
          onClick={() => setShowAll(!showAll)}
        >
          {showAll
            ? "Show threshold results"
            : `Show all ${hypotheses.length}`}
        </button>
      </div>

      {!list.length ? (
        <Empty>
          No hypotheses meet the configured threshold. You can
          inspect the complete posterior distribution using “Show
          all”.
        </Empty>
      ) : (
        <div className="findings-layout">
          <div className="findings-list">
            {list.map((h) => (
              <button
                className={`finding-row ${
                  selected === h.id ? "selected" : ""
                }`}
                key={h.id}
                onClick={() => setSelected(h.id)}
              >
                <span className="finding-id">{h.id}</span>

                <span className="finding-name">
                  <b>{h.title}</b>
                  <small>
                    {isMultiFault
                      ? h.detected
                        ? "Detected"
                        : "Not detected"
                      : h.detected
                        ? "At / above threshold"
                        : "Below threshold"}
                  </small>
                </span>

                <span className="finding-bar">
                  <i style={{ width: `${h.value * 100}%` }} />
                </span>

                <b className="finding-pct">{pct(h.value)}</b>
              </button>
            ))}
          </div>

          <div className="finding-detail">
            {current ? (
              <>
                <div className="section-kicker">
                  {current.id} · HYPOTHESIS DETAIL
                </div>

                <h3>{current.title}</h3>

                <div className="detail-big-pct">
                  {pct(current.value)}
                </div>

                {current.detail ? (
                  <>
                    <h4>Explanation</h4>
                    <ul>
                      {current.points.map((p, i) => (
                        <li key={i}>{p}</li>
                      ))}
                    </ul>

                    <h4>Recommendation</h4>
                    <p>
                      {current.recommendation ||
                        "No recommendation provided."}
                    </p>

                    <h4>Supporting evidence</h4>
                    <EvidenceList evidence={current.evidence} />
                  </>
                ) : (
                  <p className="muted-copy">
                    No detailed explanation was supplied for this
                    hypothesis. The posterior value is shown without
                    inferring additional claims.
                  </p>
                )}
              </>
            ) : (
              <div className="detail-placeholder">
                <span>↖</span>
                <b>Select a hypothesis</b>
                <p>
                  Choose a row to inspect its available explanation
                  and supporting evidence.
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      {explanation.overall_assessment && (
        <div className="assessment-copy">
          <b>Overall assessment</b>
          <p>{explanation.overall_assessment}</p>
        </div>
      )}
    </section>
  );
}

/* ============================================================
   EVIDENCE
   ============================================================ */

function EvidenceView({ evidence, hypotheses }) {
  return (
    <section className="surface page-surface">
      <div className="page-section-heading">
        <div>
          <div className="section-kicker">OBSERVATIONS</div>
          <h2>Evidence explorer</h2>
          <p>
            Evidence returned by the acquisition and diagnostic
            pipeline.
          </p>
        </div>
      </div>

      {evidence.length ? (
        <div className="evidence-cards">
          {evidence.map((item, i) => (
            <article
              className="evidence-card"
              key={`${item.evidence_type}-${i}`}
            >
              <div className="evidence-card-top">
                <b>{titleCase(item.evidence_type)}</b>
                <span
                  className={`state-chip ${String(
                    item.status
                  ).toLowerCase()}`}
                >
                  {titleCase(item.status)}
                </span>
              </div>

              <div className="evidence-source">
                Source · {item.source ?? "Not specified"}
              </div>

              {item.reason && <p>{item.reason}</p>}

              <h4>Measurements</h4>

              {Object.keys(item.measurements ?? {}).length ? (
                <dl>
                  {Object.entries(item.measurements).map(([k, v]) => (
                    <div
                      className={`measurement-row ${
                        k === "feature_shifts" ? "measurement-wide" : ""
                      }`}
                      key={k}
                    >
                      <dt>{titleCase(k)}</dt>
                      <dd>
                        {k === "feature_shifts" && v && typeof v === "object" ? (
                          <details className="feature-shifts">
                            <summary>
                              View {Object.keys(v).length} feature measurements
                            </summary>
                            <div className="feature-shift-table">
                              <table className="data-table">
                                <thead>
                                  <tr>
                                    <th>Feature</th>
                                    <th>Shift</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {Object.entries(v).map(([feature, shift]) => (
                                    <tr key={feature}>
                                      <td>{feature}</td>
                                      <td>{Number(shift).toFixed(6)}</td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          </details>
                        ) : typeof v === "object" && v !== null ? (
                          <pre className="measurement-json">
                            {JSON.stringify(v, null, 2)}
                          </pre>
                        ) : (
                          pretty(v)
                        )}
                      </dd>
                    </div>
                  ))}
                </dl>
              ) : (
                <Empty>No measurements available.</Empty>
              )}

              <details>
                <summary>Evidence metadata</summary>
                <pre>
                  {JSON.stringify(item.metadata ?? {}, null, 2)}
                </pre>
              </details>
            </article>
          ))}
        </div>
      ) : (
        <Empty>
          No acquired evidence groups were returned for this run.
        </Empty>
      )}

      <div className="subsection-heading">
        <h3>Evidence linked to hypotheses</h3>
      </div>

      {hypotheses.some((h) => h.evidence.length) ? (
        hypotheses
          .filter((h) => h.evidence.length)
          .map((h) => (
            <div className="linked-evidence" key={h.id}>
              <h4>
                {h.title} <span>{h.id}</span>
              </h4>
              <EvidenceList evidence={h.evidence} />
            </div>
          ))
      ) : (
        <Empty>
          No per-hypothesis evidence details were returned.
        </Empty>
      )}
    </section>
  );
}

/* ============================================================
   RUNTIME
   ============================================================ */

function RuntimeView({ events, runtime }) {
  return (
    <section className="surface page-surface">
      <div className="page-section-heading">
        <div>
          <div className="section-kicker">INSTRUMENTATION</div>
          <h2>Runtime execution</h2>
          <p>
            Captured operations, component timings and dataset
            references.
          </p>
        </div>

        <span className="state-chip neutral">
          {events.length} EVENTS
        </span>
      </div>

      <div className="runtime-kpis">
        <Kpi label="Successful" value={runtime.successful_events ?? 0} />
        <Kpi label="Failed" value={runtime.failed_events ?? 0} />
        <Kpi
          label="Total duration"
          value={duration(runtime.total_duration_ms)}
        />
        <Kpi
          label="Dataset references"
          value={runtime.dataset_reference_count ?? 0}
        />
      </div>

      {events.length ? (
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Event</th>
                <th>Component</th>
                <th>Operation</th>
                <th>Status</th>
                <th>Duration</th>
              </tr>
            </thead>

            <tbody>
              {events.map((e, i) => (
                <tr key={e.event_id ?? i}>
                  <td>{e.event_id ?? `E-${i + 1}`}</td>
                  <td>{e.component ?? "—"}</td>
                  <td>{e.operation ?? "—"}</td>
                  <td>
                    <span
                      className={`state-chip ${String(
                        e.status
                      ).toLowerCase()}`}
                    >
                      {titleCase(e.status)}
                    </span>
                  </td>
                  <td>{duration(e.duration_ms)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <Empty>No runtime events were captured.</Empty>
      )}

      {events.map((e, i) => (
        <details
          className="event-detail"
          key={e.event_id ?? i}
        >
          <summary>
            <b>
              {e.event_id ?? `Event ${i + 1}`} · {e.operation}
            </b>
            <span>
              {e.component} · {duration(e.duration_ms)}
            </span>
          </summary>

          <div className="event-detail-grid">
            <div>
              <label>Input references</label>
              <pre>
                {JSON.stringify(e.input_dataset_refs ?? [], null, 2)}
              </pre>
            </div>

            <div>
              <label>Output references</label>
              <pre>
                {JSON.stringify(e.output_dataset_refs ?? [], null, 2)}
              </pre>
            </div>
          </div>

          <pre>{JSON.stringify(e.metadata ?? {}, null, 2)}</pre>
        </details>
      ))}
    </section>
  );
}

/* ============================================================
   EVALUATIONS
   ============================================================ */

const metricDescriptions = {
  accuracy:
    "Percentage of examples for which the model predicted the correct target label.",
  precision:
    "Of the examples predicted as positive, the percentage that were actually positive.",
  recall:
    "Of the actual positive examples, the percentage the model correctly identified.",
  f1:
    "Harmonic mean of precision and recall. Balances the two measures.",
};

const formatMetric = (name, value) => {
  const numeric = Number(value);

  if (
    name?.toLowerCase() === "accuracy" &&
    Number.isFinite(numeric)
  ) {
    return `${(numeric * 100).toFixed(2)}%`;
  }

  return pretty(value);
};

function EvaluationView({ evaluations }) {
  return (
    <section className="surface page-surface">
      <div className="page-section-heading">
        <div>
          <div className="section-kicker">MODEL PERFORMANCE</div>
          <h2>Evaluation results</h2>
          <p>
            Only metrics explicitly recorded during this run are
            displayed.
          </p>
        </div>

        <span className="state-chip neutral">
          {evaluations.length} RECORDED
        </span>
      </div>

      {evaluations.length ? (
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Context</th>
                <th>Source</th>
              </tr>
            </thead>

            <tbody>
              {evaluations.map((e, i) => (
                <tr key={i}>
                  <td>{e.metric_name ?? "—"}</td>
                  <td>{pretty(e.metric_value)}</td>
                  <td>{e.evaluation_context ?? "—"}</td>
                  <td>{e.source ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="empty-large">
          <div className="empty-symbol">∅</div>
          <h3>No evaluation metrics recorded</h3>
          <p>
            This run captured pipeline execution events, but no model
            evaluation records were produced. Accuracy, precision,
            recall and F1 are not available for this run.
          </p>
        </div>
      )}

      {evaluations.map((e, i) => (
        <article className="evaluation-detail" key={i}>
          <div className="evaluation-info">
            <b>{titleCase(e.metric_name)}</b>
            <p>
              {metricDescriptions[e.metric_name?.toLowerCase()] ??
                "Recorded model evaluation metric."}
            </p>
            <span>
              Evaluation split: {titleCase(e.evaluation_context ?? "Unknown")}
            </span>
            <span>
              Recorded by: {titleCase(e.source ?? "Unknown")}
            </span>
          </div>

          <strong className="evaluation-value">
            {formatMetric(e.metric_name, e.metric_value)}
          </strong>
        </article>
      ))}
    </section>
  );
}

/* ============================================================
   REPORT
   ============================================================ */

function ReportView({ report }) {
  const text =
    typeof report === "string"
      ? report
      : JSON.stringify(report ?? {}, null, 2);

  return (
    <section className="surface page-surface">
      <div className="page-section-heading">
        <div>
          <div className="section-kicker">GENERATED OUTPUT</div>
          <h2>Diagnostic report</h2>
          <p>Deterministic report returned by PipeDiag.</p>
        </div>

        <button
          className="button button-quiet"
          onClick={() => navigator.clipboard?.writeText(text)}
        >
          Copy report
        </button>
      </div>

      <div className="markdown-report">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>
          {text}
        </ReactMarkdown>
      </div>
    </section>
  );
}

/* ============================================================
   SHARED COMPONENTS
   ============================================================ */

function EvidenceList({ evidence }) {
  if (!evidence?.length) {
    return <Empty>No supporting evidence listed.</Empty>;
  }

  return (
    <div className="linked-evidence-list">
      {evidence.map((item, i) => (
        <div className="linked-evidence-item" key={i}>
          <b>
            {titleCase(item.evidence_group)} · {item.metric}
          </b>

          <strong>{pretty(item.value)}</strong>

          <p>
            {item.interpretation || "No interpretation provided."}
          </p>
        </div>
      ))}
    </div>
  );
}

function Kpi({ label, value }) {
  return (
    <div className="runtime-kpi">
      <span>{label}</span>
      <b>{value}</b>
    </div>
  );
}

function Empty({ children }) {
  return <div className="empty-inline">{children}</div>;
}

export default App;