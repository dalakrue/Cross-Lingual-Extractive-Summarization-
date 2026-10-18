import { useState } from "react";

const SENTENCE_OPTIONS = [5, 7, 10];
const MAX_ARTICLE_LENGTH = 1_000_000;

function App() {
  const [article, setArticle] = useState("");
  const [sentenceCount, setSentenceCount] = useState(5);
  const [summary, setSummary] = useState("");
  const [translatedSummary, setTranslatedSummary] = useState("");
  const [displayLanguage, setDisplayLanguage] = useState("en");
  const [summarySentenceCount, setSummarySentenceCount] = useState(0);
  const [analyzedSentenceCount, setAnalyzedSentenceCount] = useState(0);
  const [cleanPreview, setCleanPreview] = useState(null);
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isTranslating, setIsTranslating] = useState(false);

  async function requestJson(path, body) {
    const response = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const result = await response.json();

    if (!response.ok) {
      const detail = Array.isArray(result.detail)
        ? result.detail.map((item) => item.msg).join(" ")
        : result.detail;
      throw new Error(detail || "The request could not be completed.");
    }

    return result;
  }

  async function handlePreviewCleanup() {
    setError("");
    setCleanPreview(null);

    if (!article.trim()) {
      setError("Paste an article before previewing text cleanup.");
      return;
    }

    if (article.length > MAX_ARTICLE_LENGTH) {
      setError("Articles must be 1,000,000 characters or fewer.");
      return;
    }

    setIsLoading(true);
    try {
      const result = await requestJson("/api/clean", { article });
      setCleanPreview(result);
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Could not preview the cleaned article.",
      );
    } finally {
      setIsLoading(false);
    }
  }

  function useCleanedText() {
    if (!cleanPreview) {
      return;
    }

    setArticle(cleanPreview.cleaned_text);
    setCleanPreview(null);
    setError("");
  }

  async function handleSubmit(event) {
    event.preventDefault();
    setError("");
    setSummary("");
    setTranslatedSummary("");
    setDisplayLanguage("en");

    if (!article.trim()) {
      setError("Paste an article before creating a summary.");
      return;
    }

    if (article.length > MAX_ARTICLE_LENGTH) {
      setError("Articles must be 1,000,000 characters or fewer.");
      return;
    }

    setIsLoading(true);
    try {
      const result = await requestJson("/api/summarize", {
        article,
        sentence_count: sentenceCount,
      });

      setSummary(result.summary);
      setSummarySentenceCount(result.sentence_count);
      setAnalyzedSentenceCount(result.analyzed_sentence_count);
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Could not connect to the summarization service. Check that the API is running.",
      );
    } finally {
      setIsLoading(false);
    }
  }

  async function toggleSummaryLanguage(language) {
    setError("");
    if (language === "en") {
      setDisplayLanguage("en");
      return;
    }

    if (translatedSummary) {
      setDisplayLanguage("my");
      return;
    }

    setIsTranslating(true);
    try {
      const result = await requestJson("/api/translate", { text: summary });
      setTranslatedSummary(result.translated_text);
      setDisplayLanguage("my");
    } catch (translationError) {
      setError(
        translationError instanceof Error
          ? translationError.message
          : "Could not translate the summary into Myanmar.",
      );
    } finally {
      setIsTranslating(false);
    }
  }

  async function copySummary() {
    try {
      await navigator.clipboard.writeText(
        displayLanguage === "my" ? translatedSummary : summary,
      );
    } catch {
      setError("The summary could not be copied from this browser.");
    }
  }

  return (
    <main className="page-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="Briefly home">
          <span className="brand-mark" aria-hidden="true">
            B
          </span>
          <span>briefly</span>
        </a>
        <span className="model-label">
          <span className="status-dot" />
          NLP article summarizer
        </span>
      </header>

      <section className="intro">
        <p className="eyebrow">Read less. Keep the important parts.</p>
        <h1>Your article,<br />made <span>brief.</span></h1>
        <p className="intro-copy">
          Paste an English article and get its key sentences, selected by the
          project&apos;s trained NLP model.
        </p>
      </section>

      <section className="workspace" aria-label="Article summarizer">
        <form className="input-card" onSubmit={handleSubmit}>
          <div className="card-heading">
            <div>
              <span className="step-number">01</span>
              <h2>Article text</h2>
            </div>
            <span className="character-count">
              {article.length.toLocaleString()} / 1,000,000 characters
            </span>
          </div>
          <label className="visually-hidden" htmlFor="article">
            Paste your article
          </label>
          <textarea
            id="article"
            value={article}
            onChange={(event) => {
              setArticle(event.target.value);
              setCleanPreview(null);
            }}
            placeholder="Paste your article here..."
            rows={12}
          />

          <div className="cleanup-actions">
            <p>Clean copy/paste artifacts and preview the text before summarizing.</p>
            <button
              className="copy-button"
              type="button"
              onClick={handlePreviewCleanup}
              disabled={isLoading}
            >
              {isLoading ? "Preparing preview..." : "Preview cleaned text"}
            </button>
          </div>

          {cleanPreview && (
            <section className="clean-preview" aria-label="Cleaned article preview">
              <div className="preview-heading">
                <div>
                  <h3>Review cleaned article</h3>
                  <p>
                    {cleanPreview.paragraph_count} paragraphs ·{" "}
                    {cleanPreview.sentence_count} sentences ·{" "}
                    {cleanPreview.original_character_count.toLocaleString()} →{" "}
                    {cleanPreview.cleaned_character_count.toLocaleString()} characters
                  </p>
                </div>
                <button
                  className="copy-button"
                  type="button"
                  onClick={useCleanedText}
                >
                  Use cleaned text
                </button>
              </div>
              <pre className="preview-text">{cleanPreview.cleaned_text}</pre>
            </section>
          )}

          <div className="form-footer">
            <fieldset className="sentence-picker">
              <legend>Summary length</legend>
              <div className="option-list">
                {SENTENCE_OPTIONS.map((option) => (
                  <label className="option" key={option}>
                    <input
                      type="radio"
                      name="sentenceCount"
                      value={option}
                      checked={sentenceCount === option}
                      onChange={() => setSentenceCount(option)}
                    />
                    <span>{option} sentences</span>
                  </label>
                ))}
              </div>
            </fieldset>
            <button
              className="summarize-button"
              type="submit"
              disabled={isLoading}
            >
              {isLoading ? "Summarizing..." : "Summarize article"}
              {!isLoading && <span aria-hidden="true">↗</span>}
            </button>
          </div>
        </form>

        <section
          className={`output-card${summary ? " has-summary" : ""}`}
          aria-live="polite"
          aria-label="Summary output"
        >
          <div className="card-heading output-heading">
            <div>
              <span className="step-number">02</span>
              <h2>Summary</h2>
            </div>
            {summary && (
              <div className="summary-actions">
                <div className="language-toggle" aria-label="Summary language">
                  <button
                    className={displayLanguage === "en" ? "selected" : ""}
                    type="button"
                    aria-pressed={displayLanguage === "en"}
                    onClick={() => toggleSummaryLanguage("en")}
                    disabled={isTranslating}
                  >
                    English
                  </button>
                  <button
                    className={displayLanguage === "my" ? "selected" : ""}
                    type="button"
                    aria-pressed={displayLanguage === "my"}
                    onClick={() => toggleSummaryLanguage("my")}
                    disabled={isTranslating || !summary}
                  >
                    မြန်မာ
                  </button>
                </div>
                <button className="copy-button" type="button" onClick={copySummary}>
                  Copy
                </button>
              </div>
            )}
          </div>
          {error && <p className="message error-message" role="alert">{error}</p>}
          {summary ? (
            <>
              <p className="translation-notice">
                Choosing မြန်မာ sends only this generated summary to MyMemory,
                a free third-party translation service. Free-tier availability
                and machine translation quality may vary.
              </p>
              {isTranslating ? (
                <p className="translation-loading" role="status">
                  Translating summary into Myanmar...
                </p>
              ) : (
                <p className="summary-text">
                  {displayLanguage === "my" ? translatedSummary : summary}
                </p>
              )}
              <p className="summary-meta">
                Analyzed {analyzedSentenceCount} sentences · Selected{" "}
                {summarySentenceCount} {summarySentenceCount === 1 ? "sentence" : "sentences"}
              </p>
            </>
          ) : !error ? (
            <div className="empty-state">
              <span className="empty-icon" aria-hidden="true">✳</span>
              <p>Your summary will appear here.</p>
              <span>Choose a length and summarize your article.</span>
            </div>
          ) : null}
        </section>
      </section>

      <footer className="page-footer">
        <span>Extractive summarization</span>
        <span>Powered by the saved FastText + SVM models</span>
      </footer>
    </main>
  );
}

export default App;
