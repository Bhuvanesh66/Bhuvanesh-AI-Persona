'use client'

import { useState, useRef, useEffect, KeyboardEvent } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

// ── types ─────────────────────────────────────────────────────────────────────

interface Source {
  n: number
  repo: string | null
  file: string | null
  type: string
  is_fork: boolean
  title: string
}

interface Message {
  id: number
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
  streaming?: boolean
}

// ── SSE parser ────────────────────────────────────────────────────────────────

function parseSSEBlock(block: string): { type: string; data: string } | null {
  const lines = block.split('\n')
  let type = 'message'
  let data = ''
  for (const line of lines) {
    if (line.startsWith('event: ')) type = line.slice(7).trim()
    else if (line.startsWith('data: ')) data = line.slice(6).trim()
  }
  return data ? { type, data } : null
}

// ── constants ─────────────────────────────────────────────────────────────────

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'
const CALCOM_URL = process.env.NEXT_PUBLIC_CALCOM_URL ?? ''

const WELCOME: Message = {
  id: 0,
  role: 'assistant',
  content:
    "Hi! I'm Bhuvanesh's AI representative. Ask me anything about his background, skills, or projects — or book a 30-min call to connect with him directly.",
}

const SUGGESTIONS = [
  'What are your technical skills?',
  'Tell me about your projects',
  "What's your educational background?",
  'Are you open to internship opportunities?',
]

// ── main component ────────────────────────────────────────────────────────────

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([WELCOME])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [showSuggestions, setShowSuggestions] = useState(true)
  const nextId = useRef(1)
  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  function buildHistory() {
    return messages
      .filter((m) => !m.streaming && m.id !== 0)
      .slice(-12)
      .map((m) => ({ role: m.role, content: m.content }))
  }

  async function send(text?: string) {
    const q = (text ?? input).trim()
    if (!q || loading) return
    setInput('')
    setLoading(true)
    setShowSuggestions(false)

    const userId = nextId.current++
    const assistantId = nextId.current++

    setMessages((prev) => [
      ...prev,
      { id: userId, role: 'user', content: q },
      { id: assistantId, role: 'assistant', content: '', streaming: true },
    ])

    try {
      const res = await fetch(`${API_URL}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: q, history: buildHistory() }),
      })

      if (!res.ok || !res.body) throw new Error(`API ${res.status}`)

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buf = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buf += decoder.decode(value, { stream: true })

        const parts = buf.split('\n\n')
        buf = parts.pop() ?? ''

        for (const part of parts) {
          const ev = parseSSEBlock(part)
          if (!ev || ev.data === '{}') continue
          try {
            const payload = JSON.parse(ev.data)
            if (payload.delta !== undefined) {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId
                    ? { ...m, content: m.content + payload.delta }
                    : m
                )
              )
            } else if (payload.sources) {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantId ? { ...m, sources: payload.sources } : m
                )
              )
            }
          } catch {
            // skip malformed JSON
          }
        }
      }
    } catch {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId
            ? {
                ...m,
                content:
                  'Something went wrong — the server may be waking up. Please try again in a moment.',
                streaming: false,
              }
            : m
        )
      )
    } finally {
      setMessages((prev) =>
        prev.map((m) => (m.id === assistantId ? { ...m, streaming: false } : m))
      )
      setLoading(false)
      inputRef.current?.focus()
    }
  }

  function onKey(e: KeyboardEvent<HTMLInputElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      {/* ── header ── */}
      <header
        className="relative flex items-center gap-4 px-5 py-4 shadow-sm"
        style={{ background: 'linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)' }}
      >
        {/* avatar */}
        <div className="w-11 h-11 rounded-full bg-white/20 backdrop-blur flex items-center justify-center text-white font-bold text-base flex-shrink-0 ring-2 ring-white/30">
          BM
        </div>

        {/* name + title */}
        <div className="flex-1 min-w-0">
          <p className="font-semibold text-white text-base leading-tight">Bhuvanesh M S</p>
          <div className="flex items-center gap-1.5 mt-0.5">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 flex-shrink-0" />
            <p className="text-xs text-indigo-200 truncate">
              AI Engineer · Scaler School of Technology · BITS Pilani
            </p>
          </div>
        </div>

        {/* book a call */}
        {CALCOM_URL && (
          <a
            href={CALCOM_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="flex-shrink-0 text-xs font-semibold bg-white text-indigo-700 px-4 py-2 rounded-full
                       hover:bg-indigo-50 transition-colors shadow-sm"
          >
            Book a Call
          </a>
        )}
      </header>

      {/* ── messages ── */}
      <main className="flex-1 overflow-y-auto">
        <div className="max-w-2xl mx-auto px-4 py-6 space-y-5">
          {messages.map((msg) => (
            <MessageBubble key={msg.id} msg={msg} />
          ))}

          {/* suggestion chips — visible until first user message */}
          {showSuggestions && !loading && (
            <div className="flex flex-wrap gap-2 pl-9">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => send(s)}
                  className="text-xs px-3 py-1.5 rounded-full border border-indigo-200 text-indigo-600
                             bg-white hover:bg-indigo-50 hover:border-indigo-400 transition-colors shadow-sm"
                >
                  {s}
                </button>
              ))}
            </div>
          )}

          <div ref={bottomRef} />
        </div>
      </main>

      {/* ── input bar ── */}
      <footer className="bg-white border-t border-gray-200 px-4 py-3">
        <div className="max-w-2xl mx-auto flex gap-2 items-center">
          <input
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKey}
            disabled={loading}
            placeholder="Ask about Bhuvanesh's skills, projects, or experience…"
            className="flex-1 rounded-2xl border border-gray-200 bg-gray-50 px-4 py-3 text-sm
                       outline-none focus:ring-2 focus:ring-indigo-400 focus:border-transparent
                       focus:bg-white transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          />
          <button
            onClick={() => send()}
            disabled={loading || !input.trim()}
            className="rounded-2xl bg-indigo-600 px-5 py-3 text-sm font-semibold text-white
                       transition-all hover:bg-indigo-700 active:scale-95
                       disabled:bg-gray-200 disabled:text-gray-400 disabled:cursor-not-allowed
                       flex-shrink-0 shadow-sm"
          >
            {loading ? (
              <span className="flex items-center gap-1.5">
                <span className="w-1 h-1 bg-white/60 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                <span className="w-1 h-1 bg-white/60 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                <span className="w-1 h-1 bg-white/60 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
              </span>
            ) : (
              'Send'
            )}
          </button>
        </div>
        <p className="text-center text-[10px] text-gray-300 mt-2">
          AI-generated responses · May not reflect real-time information
        </p>
      </footer>
    </div>
  )
}

// ── avatar ────────────────────────────────────────────────────────────────────

function Avatar() {
  return (
    <div
      className="w-8 h-8 rounded-full flex items-center justify-center text-white text-xs
                 font-bold flex-shrink-0 self-end"
      style={{ background: 'linear-gradient(135deg, #4f46e5, #7c3aed)' }}
    >
      BM
    </div>
  )
}

// ── message bubble ────────────────────────────────────────────────────────────

function MessageBubble({ msg }: { msg: Message }) {
  const [open, setOpen] = useState(false)
  const isUser = msg.role === 'user'

  return (
    <div className={`flex gap-2.5 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
      {!isUser && <Avatar />}

      <div
        className={`max-w-[78%] flex flex-col gap-1 ${isUser ? 'items-end' : 'items-start'}`}
      >
        {/* bubble */}
        <div
          className={`rounded-2xl px-4 py-3 text-sm leading-relaxed min-h-[44px] ${
            isUser
              ? 'bg-indigo-600 text-white rounded-br-sm whitespace-pre-wrap'
              : 'bg-white text-gray-800 border border-gray-100 shadow-sm rounded-bl-sm'
          }`}
        >
          {isUser ? (
            msg.content
          ) : (
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                ul: ({ children }) => <ul className="list-disc pl-4 mb-2 space-y-0.5">{children}</ul>,
                ol: ({ children }) => <ol className="list-decimal pl-4 mb-2 space-y-0.5">{children}</ol>,
                li: ({ children }) => <li className="leading-relaxed">{children}</li>,
                strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
                code: ({ children }) => <code className="bg-gray-100 px-1 py-0.5 rounded text-xs font-mono">{children}</code>,
                pre: ({ children }) => <pre className="bg-gray-100 p-2 rounded text-xs overflow-x-auto mb-2">{children}</pre>,
                h1: ({ children }) => <h1 className="font-bold text-base mb-1">{children}</h1>,
                h2: ({ children }) => <h2 className="font-bold text-sm mb-1">{children}</h2>,
                h3: ({ children }) => <h3 className="font-semibold text-sm mb-1">{children}</h3>,
              }}
            >
              {msg.content}
            </ReactMarkdown>
          )}

          {/* typing dots when waiting for first token */}
          {msg.streaming && !msg.content && (
            <span className="inline-flex items-center gap-1 h-4">
              <span className="w-2 h-2 bg-gray-300 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
              <span className="w-2 h-2 bg-gray-300 rounded-full animate-bounce" style={{ animationDelay: '160ms' }} />
              <span className="w-2 h-2 bg-gray-300 rounded-full animate-bounce" style={{ animationDelay: '320ms' }} />
            </span>
          )}

          {/* blinking cursor while streaming text */}
          {msg.streaming && msg.content && (
            <span className="ml-0.5 inline-block w-0.5 h-3.5 bg-gray-400 animate-pulse align-middle" />
          )}
        </div>

        {/* sources */}
        {msg.sources && msg.sources.length > 0 && (
          <div className="w-full">
            <button
              onClick={() => setOpen((v) => !v)}
              className="text-[11px] text-gray-400 hover:text-indigo-500 transition-colors"
            >
              {open
                ? '▾ hide sources'
                : `▸ ${msg.sources.length} source${msg.sources.length > 1 ? 's' : ''}`}
            </button>
            {open && (
              <ul className="mt-1 pl-2 border-l-2 border-indigo-100 space-y-1">
                {msg.sources.map((s) => (
                  <li key={s.n} className="flex items-center gap-1.5 text-[11px] text-gray-400">
                    <span className="font-mono text-gray-300">[{s.n}]</span>
                    <span className="truncate">{s.title}</span>
                    {s.is_fork && (
                      <span className="shrink-0 text-[9px] border border-amber-300 text-amber-500 px-1 rounded">
                        fork
                      </span>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
