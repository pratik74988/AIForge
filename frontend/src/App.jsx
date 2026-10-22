
import { useState } from "react";
import {
  Activity, AlertTriangle, ArrowRight, Check, CheckCircle2,
  ChevronRight, CircleDot, CloudUpload, Cpu, Database, FileText,
  FlaskConical, Gauge, GitBranch, LoaderCircle, Play, ShieldCheck,
  Sparkles, Workflow, X
} from "lucide-react";
import "./App.css";

const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

async function request(path, options = {}) {
  const response = await fetch(`${API}${path}`, options);
  const body = await response.json().catch(() => ({}));

  if (!response.ok) {
    const detail = body.detail;
    throw new Error(
      typeof detail === "string"
        ? detail
        : Array.isArray(detail)
          ? detail.map((item) => item.msg || JSON.stringify(item)).join("; ")
          : "Request failed"
    );
  }
  return body;
}

function Panel({ title, icon: Icon, children, right }) {
  return (
    <section className="panel">
      <div className="panel-heading">
        <div className="panel-title"><Icon size={17} /><span>{title}</span></div>
        {right}
      </div>
      {children}
    </section>
  );
}

function JsonView({ value }) {
  return <pre className="json-view">{JSON.stringify(value, null, 2)}</pre>;
}

function Status({ ok, children }) {
  return (
    <span className={`status ${ok ? "status-ok" : "status-warn"}`}>
      {ok ? <CheckCircle2 size={13} /> : <AlertTriangle size={13} />}
      {children}
    </span>
  );
}

export default function App() {
  const [file, setFile] = useState(null);
  const [target, setTarget] = useState("");
  const [requirement, setRequirement] = useState(
    "Build a reliable machine learning model to predict the target column."
  );
  const [profile, setProfile] = useState(null);
  const [plan, setPlan] = useState(null);
  const [compliance, setCompliance] = useState(null);
  const [compiled, setCompiled] = useState(null);
  const [approved, setApproved] = useState(false);
  const [activeStep, setActiveStep] = useState(0);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [view, setView] = useState("overview");

  async function run(label, action) {
    setBusy(label);
    setError("");
    try {
      await action();
    } catch (err) {
      setError(err.message || "Something went wrong.");
    } finally {
      setBusy("");
    }
  }

  function resetDownstream() {
    setPlan(null);
    setCompliance(null);
    setCompiled(null);
    setApproved(false);
  }

  function analyze() {
    if (!file) {
      setError("Choose a CSV file first.");
      return;
    }
    return run("analyze", async () => {
      const form = new FormData();
      form.append("file", file);
      if (target.trim()) form.append("target", target.trim());

      const result = await request("/api/analyze", {
        method: "POST",
        body: form,
      });

      setProfile(result);
      resetDownstream();
      setActiveStep(1);
      setView("overview");
    });
  }

  function createPlan() {
    if (!profile) return;
    return run("plan", async () => {
      const result = await request("/api/pipeline/plan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_profile: profile,
          user_requirement: requirement,
        }),
      });

      setPlan(result);
      setCompliance(null);
      setCompiled(null);
      setApproved(false);
      setActiveStep(2);
    });
  }

  function validatePlan() {
    if (!plan?.proposed_spec) return;
    return run("validate", async () => {
      const result = await request("/api/pipeline/validate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ spec: plan.proposed_spec }),
      });

      setCompliance(result);
      setCompiled(null);
      setActiveStep(3);
    });
  }

  function compilePlan() {
    if (!plan?.proposed_spec || !approved) return;
    return run("compile", async () => {
      const result = await request("/api/pipeline/compile", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          spec: plan.proposed_spec,
          user_approved: true,
        }),
      });

      setCompiled(result);
      setActiveStep(4);
    });
  }

  const columns = profile?.columns || [];
  const spec = plan?.proposed_spec;
  const compliancePassed = compliance?.approved === true;
  const needsHumanApproval = compliance?.requires_human_approval === true;
  const canCompile = compliancePassed && !needsHumanApproval && approved;

  const steps = [
    { label: "Dataset", icon: Database },
    { label: "Planning", icon: Sparkles },
    { label: "Compliance", icon: ShieldCheck },
    { label: "Compilation", icon: Cpu },
  ];

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><Workflow size={22} /></div>
          <div><strong>AI<span>Forge</span></strong><small>ML WORKFLOW STUDIO</small></div>
        </div>

        <div className="workspace-label">WORKSPACE</div>
        <button className={`nav-item ${view === "overview" ? "selected" : ""}`}
          onClick={() => setView("overview")}>
          <Gauge size={17} /> Overview
        </button>
        <button className={`nav-item ${view === "artifacts" ? "selected" : ""}`}
          onClick={() => setView("artifacts")}>
          <GitBranch size={17} /> Pipeline artifacts
        </button>

        <div className="sidebar-divider" />
        <div className="workspace-label">YOUR PIPELINE</div>
        {steps.map((step, index) => {
          const Icon = step.icon;
          const done = index === 0 ? !!profile
            : index === 1 ? !!plan
            : index === 2 ? !!compliance
            : !!compiled;
          return (
            <div className="side-step" key={step.label}>
              <div className={`side-step-icon ${done ? "done" : ""}`}>
                {done ? <Check size={14} /> : <Icon size={14} />}
              </div>
              <span>{step.label}</span>
              {done && <span className="tiny-check">●</span>}
            </div>
          );
        })}

        <div className="sidebar-bottom">
          <div className="online-dot" />
          <div><strong>Local environment</strong><small>API-driven · No training yet</small></div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumb">Workspace <ChevronRight size={14} /> <strong>{view === "overview" ? "Pipeline overview" : "Artifacts"}</strong></div>
          <div className="topbar-right">
            <span className="environment"><span /> DEVELOPMENT</span>
            <div className="avatar">AF</div>
          </div>
        </header>

        <div className="page-content">
          <div className="page-intro">
            <div>
              <div className="eyebrow"><CircleDot size={12} /> INTELLIGENT ML WORKFLOW</div>
              <h1>{view === "overview" ? "Build your ML pipeline" : "Pipeline artifacts"}</h1>
              <p>From raw data to a validated, compiled pipeline. One deliberate step at a time.</p>
            </div>
            <div className="run-indicator"><span /> API WORKSPACE</div>
          </div>

          {error && (
            <div className="error-banner">
              <AlertTriangle size={17} />
              <span>{error}</span>
              <button onClick={() => setError("")}><X size={16} /></button>
            </div>
          )}

          {view === "overview" ? (
            <>
              <div className="workflow-strip">
                {steps.map((step, i) => {
                  const Icon = step.icon;
                  const done = i === 0 ? !!profile : i === 1 ? !!plan : i === 2 ? !!compliance : !!compiled;
                  return (
                    <div className="workflow-step" key={step.label}>
                      <div className={`workflow-icon ${done ? "done" : ""} ${activeStep === i + 1 ? "current" : ""}`}>
                        {done ? <Check size={17} /> : <Icon size={17} />}
                      </div>
                      <div className="workflow-step-text">
                        <small>STEP 0{i + 1}</small>
                        <strong>{step.label}</strong>
                      </div>
                      {i < steps.length - 1 && <div className={`workflow-line ${done ? "line-done" : ""}`} />}
                    </div>
                  );
                })}
              </div>

              <div className="stats-grid">
                <div className="stat-card">
                  <div className="stat-icon purple"><Database size={18} /></div>
                  <div><span>DATASET ROWS</span><strong>{profile?.dataset?.rows?.toLocaleString() ?? "—"}</strong><small>{profile ? "Analyzed dataset" : "Awaiting upload"}</small></div>
                </div>
                <div className="stat-card">
                  <div className="stat-icon blue"><GitBranch size={18} /></div>
                  <div><span>FEATURE COLUMNS</span><strong>{profile ? columns.length : "—"}</strong><small>{profile ? "Detected columns" : "Awaiting analysis"}</small></div>
                </div>
                <div className="stat-card">
                  <div className="stat-icon green"><ShieldCheck size={18} /></div>
                  <div><span>POLICY STATUS</span><strong className="stat-status">{!compliance ? "Pending" : compliancePassed ? needsHumanApproval ? "Review" : "Passed" : "Blocked"}</strong><small>{!compliance ? "Not validated yet" : `${(compliance.errors || []).length} error(s)`}</small></div>
                </div>
                <div className="stat-card">
                  <div className="stat-icon orange"><Cpu size={18} /></div>
                  <div><span>PIPELINE STATUS</span><strong className="stat-status">{compiled ? "Compiled" : "Draft"}</strong><small>{compiled ? "Ready for future execution" : "Not compiled"}</small></div>
                </div>
              </div>

              <div className="content-grid">
                <div className="left-column">
                  <Panel title="01 / Dataset ingestion" icon={Database}
                    right={profile && <Status ok>Analyzed</Status>}>
                    <p className="panel-description">Upload a CSV to profile its schema, quality, statistics, and relationships.</p>
                    <label className="field-label">DATASET FILE</label>
                    <label className="upload-zone">
                      <input type="file" accept=".csv,text/csv" onChange={(e) => {
                        setFile(e.target.files?.[0] || null);
                        setProfile(null);
                        resetDownstream();
                      }} />
                      <div className="upload-icon"><CloudUpload size={22} /></div>
                      <strong>{file ? file.name : "Drop your CSV here or browse"}</strong>
                      <span>{file ? `${(file.size / 1024).toFixed(1)} KB · Selected` : "CSV files only · Processed locally by your API"}</span>
                    </label>
                    <label className="field-label">TARGET COLUMN <span className="optional">OPTIONAL</span></label>
                    <input className="text-input" value={target} onChange={(e) => setTarget(e.target.value)}
                      placeholder="e.g. outcome, species, price" />
                    <button className="primary-button full-button" disabled={!file || !!busy} onClick={analyze}>
                      {busy === "analyze" ? <LoaderCircle className="spin" size={17} /> : <Activity size={17} />}
                      {busy === "analyze" ? "Analyzing dataset..." : "Analyze dataset"}
                      {!busy && <ArrowRight size={16} />}
                    </button>
                    {profile && (
                      <div className="success-note"><CheckCircle2 size={15} /> Dataset profile created successfully.</div>
                    )}
                  </Panel>

                  <Panel title="02 / Planning requirements" icon={Sparkles}
                    right={plan && <Status ok>Plan generated</Status>}>
                    <p className="panel-description">Describe the objective. The Planner proposes a TrainingSpec using the dataset profile.</p>
                    <label className="field-label">YOUR ML OBJECTIVE</label>
                    <textarea className="text-input requirement-input" value={requirement}
                      onChange={(e) => setRequirement(e.target.value)}
                      placeholder="Describe what you want to predict or discover..." />
                    <button className="primary-button full-button" disabled={!profile || !requirement.trim() || !!busy} onClick={createPlan}>
                      {busy === "plan" ? <LoaderCircle className="spin" size={17} /> : <Sparkles size={17} />}
                      {busy === "plan" ? "Generating plan..." : "Generate proposed plan"}
                      {!busy && <ArrowRight size={16} />}
                    </button>
                    {!profile && <p className="hint">Analyze a dataset first to unlock planning.</p>}
                    {plan && <div className="success-note"><CheckCircle2 size={15} /> Proposal generated. Review it before approval.</div>}
                  </Panel>
                </div>

                <div className="right-column">
                  <Panel title="Dataset intelligence" icon={FileText}
                    right={profile && <span className="muted-count">{columns.length} columns</span>}>
                    {!profile ? (
                      <div className="empty-state">
                        <div className="empty-icon"><Database size={23} /></div>
                        <strong>No dataset profile yet</strong>
                        <span>Upload and analyze a CSV to see its structure and quality.</span>
                      </div>
                    ) : (
                      <>
                        <div className="dataset-summary">
                          <div><small>ROWS</small><strong>{profile.dataset?.rows?.toLocaleString()}</strong></div>
                          <div><small>COLUMNS</small><strong>{profile.dataset?.columns}</strong></div>
                          <div><small>DUPLICATES</small><strong>{profile.quality?.duplicate_rows ?? "—"}</strong></div>
                        </div>
                        <div className="table-wrap">
                          <table className="data-table">
                            <thead><tr><th>Column</th><th>Type</th><th>Missing</th></tr></thead>
                            <tbody>
                              {columns.slice(0, 8).map((column) => (
                                <tr key={column.name}>
                                  <td title={column.name}>{column.name}</td>
                                  <td><span className="type-pill">{String(column.semantic_type ?? column.raw_dtype ?? "unknown")}</span></td>
                                  <td>{column.missing_percentage == null ? "—" : `${Number(column.missing_percentage).toFixed(1)}%`}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                          {columns.length > 8 && <div className="table-footer">Showing 8 of {columns.length} columns</div>}
                        </div>
                        {profile.warnings?.length > 0 && (
                          <div className="warning-box"><AlertTriangle size={15} /><span>{profile.warnings.length} dataset warning(s) detected. Review the profile artifact.</span></div>
                        )}
                      </>
                    )}
                  </Panel>

                  <Panel title="Compliance & governance" icon={ShieldCheck}
                    right={compliance && <Status ok={compliancePassed && !needsHumanApproval}>{compliancePassed ? needsHumanApproval ? "Review needed" : "Passed" : "Blocked"}</Status>}>
                    {!compliance ? (
                      <div className="empty-state compact">
                        <div className="empty-icon"><ShieldCheck size={22} /></div>
                        <strong>Awaiting proposed plan</strong>
                        <span>Validate the TrainingSpec against policy before compilation.</span>
                      </div>
                    ) : (
                      <>
                        <div className={`compliance-banner ${compliancePassed && !needsHumanApproval ? "pass" : "fail"}`}>
                          {compliancePassed && !needsHumanApproval ? <CheckCircle2 size={18} /> : <AlertTriangle size={18} />}
                          <div><strong>{compliancePassed ? needsHumanApproval ? "Additional review required" : "Policy checks passed" : "Compilation blocked"}</strong>
                          <span>{(compliance.errors || []).length} errors · {(compliance.warnings || []).length} warnings</span></div>
                        </div>
                        {[...(compliance.errors || []), ...(compliance.warnings || [])].slice(0, 5).map((item, i) => (
                          <div className="violation" key={i}>
                            <AlertTriangle size={14} /><span>{item.message || item.description || item.code || JSON.stringify(item)}</span>
                          </div>
                        ))}
                        {needsHumanApproval && <p className="hint">Compliance policy requires additional human review. This demo cannot bypass that requirement.</p>}
                      </>
                    )}
                    <button className="secondary-button full-button" disabled={!plan?.proposed_spec || !!busy} onClick={validatePlan}>
                      {busy === "validate" ? <LoaderCircle className="spin" size={16} /> : <ShieldCheck size={16} />}
                      {busy === "validate" ? "Validating..." : "Run compliance checks"}
                    </button>
                  </Panel>
                </div>
              </div>

              <Panel title="03 / Proposed TrainingSpec" icon={Workflow}
                right={spec && <Status ok>Review required</Status>}>
                {!spec ? (
                  <div className="empty-state compact">
                    <div className="empty-icon"><Workflow size={22} /></div>
                    <strong>No proposed specification</strong>
                    <span>Generate a plan to inspect the model, preprocessing, split, and evaluation choices.</span>
                  </div>
                ) : (
                  <>
                    <div className="spec-summary">
                      <div><small>TASK</small><strong>{spec.task?.type || "—"}</strong></div>
                      <div><small>MODEL TEMPLATE</small><strong>{spec.model?.template || "—"}</strong></div>
                      <div><small>ALGORITHM</small><strong>{spec.model?.algorithm || "—"}</strong></div>
                      <div><small>PRIMARY METRIC</small><strong>{spec.evaluation?.primary_metric || "—"}</strong></div>
                    </div>
                    {plan.recommendations?.length > 0 && (
                      <div className="recommendations">
                        <h4><Sparkles size={15} /> Planner recommendations</h4>
                        {plan.recommendations.map((item, i) => (
                          <div className="recommendation" key={i}>
                            <span className="recommendation-dot">{i + 1}</span>
                            <div><strong>{item.title || item.name || item.category || `Recommendation ${i + 1}`}</strong>
                            <p>{item.description || item.rationale || item.reason || JSON.stringify(item)}</p></div>
                          </div>
                        ))}
                      </div>
                    )}
                    <details className="artifact-details">
                      <summary>Inspect full TrainingSpec JSON <ChevronRight size={15} /></summary>
                      <JsonView value={spec} />
                    </details>
                    <div className="approval-row">
                      <label className="approval-check">
                        <input type="checkbox" checked={approved} onChange={(e) => {
                          setApproved(e.target.checked);
                          setCompiled(null);
                        }} />
                        <span>I have reviewed and explicitly approve this proposed specification.</span>
                      </label>
                      <button className="primary-button" disabled={!canCompile || !!busy} onClick={compilePlan}>
                        {busy === "compile" ? <LoaderCircle className="spin" size={16} /> : <Play size={16} />}
                        {busy === "compile" ? "Compiling..." : "Compile pipeline"}
                      </button>
                    </div>
                    {!compliance && <p className="hint">Run compliance checks before compilation.</p>}
                    {compliance && !compliancePassed && <p className="hint">Resolve compliance errors before compiling.</p>}
                    {needsHumanApproval && <p className="hint">Additional policy approval is required. The compile action remains blocked.</p>}
                  </>
                )}
              </Panel>

              {compiled && (
                <Panel title="04 / Compiled pipeline" icon={CheckCircle2} right={<Status ok>Compiled</Status>}>
                  <div className="compiled-banner">
                    <CheckCircle2 size={21} />
                    <div><strong>Pipeline compiled successfully</strong><span>The compiler returned a pipeline artifact. Model training has not been run.</span></div>
                  </div>
                  <JsonView value={compiled} />
                </Panel>
              )}
            </>
          ) : (
            <Panel title="Pipeline artifacts" icon={GitBranch}>
              <p className="panel-description">Inspect the outputs produced by each existing AIForge module.</p>
              {[
                ["DatasetProfile", profile],
                ["PlannerResult", plan],
                ["ComplianceResult", compliance],
                ["CompiledPipeline summary", compiled],
              ].map(([name, artifact]) => (
                <details className="artifact-details artifact-row" key={name}>
                  <summary>{name}<span className="artifact-state">{artifact ? "Available" : "Not generated"}</span><ChevronRight size={15} /></summary>
                  {artifact ? <JsonView value={artifact} /> : <p className="hint">Run the corresponding workflow step to generate this artifact.</p>}
                </details>
              ))}
            </Panel>
          )}

          <footer className="footer">
            <span>AIForge <span className="footer-dot">·</span> ML workflow prototype</span>
            <span><span className="online-dot" /> Local API connection</span>
          </footer>
        </div>
      </main>
    </div>
  );
}
