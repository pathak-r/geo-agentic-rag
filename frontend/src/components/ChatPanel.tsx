import { useState } from "react";
import { postChat, type ChatMessage } from "../api";

export function ChatPanel() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [lastSources, setLastSources] = useState<{ doc_type: string; excerpt: string }[]>([]);

  async function send() {
    const text = input.trim();
    if (!text || busy) return;
    setInput("");
    setErr(null);
    setLastSources([]);
    const prior = messages;
    const next: ChatMessage[] = [...prior, { role: "user", content: text }];
    setMessages(next);
    setBusy(true);
    try {
      const { response, sources } = await postChat(text, prior);
      setMessages([...next, { role: "assistant", content: response }]);
      setLastSources(sources || []);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Chat failed");
      setMessages(prior);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="max-w-3xl flex flex-col gap-4 h-[min(70vh,640px)]">
      <p className="text-sm text-slate-400">
        Ask about production, documents, decline, recovery, or anomalies. Follow-ups use this thread.
      </p>
      <div className="flex-1 overflow-y-auto rounded-lg border border-slate-800 bg-slate-900/40 p-4 space-y-4">
        {messages.length === 0 && (
          <p className="text-slate-600 text-sm">Try: &quot;What anomalies were detected across all wells?&quot;</p>
        )}
        {messages.map((m, i) => (
          <div
            key={i}
            className={`rounded-lg px-3 py-2 text-sm max-w-[95%] whitespace-pre-wrap ${
              m.role === "user"
                ? "ml-auto bg-sky-900/40 border border-sky-800 text-sky-100"
                : "mr-auto bg-slate-800/80 border border-slate-700 text-slate-200"
            }`}
          >
            {m.content}
          </div>
        ))}
        {busy && <p className="text-slate-500 text-sm">Thinking…</p>}
      </div>

      {lastSources.length > 0 && (
        <details className="text-xs text-slate-400">
          <summary className="cursor-pointer text-sky-400">Sources</summary>
          <ul className="mt-2 space-y-2 list-disc pl-4">
            {lastSources.map((s, i) => (
              <li key={i} className="text-slate-500">
                <span className="text-slate-400">{s.doc_type}</span>: {s.excerpt.slice(0, 200)}…
              </li>
            ))}
          </ul>
        </details>
      )}

      {err && <p className="text-red-400 text-sm">{err}</p>}

      <div className="flex gap-2">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && (e.preventDefault(), send())}
          placeholder="Ask about the Volve field…"
          disabled={busy}
          className="flex-1 rounded-lg bg-slate-900 border border-slate-700 px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-sky-500"
        />
        <button
          type="button"
          onClick={send}
          disabled={busy || !input.trim()}
          className="rounded-lg bg-sky-600 px-4 py-2 text-sm font-medium text-white disabled:opacity-40"
        >
          Send
        </button>
      </div>
    </div>
  );
}
