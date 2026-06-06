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
  { label: 'Technical Skills', prompt: 'What are your technical skills?' },
  { label: 'Projects', prompt: 'Tell me about your projects' },
  { label: 'Education', prompt: "What's your educational background?" },
  { label: 'Open to Work?', prompt: 'Are you open to internship opportunities?' },
]

// ── icon components ───────────────────────────────────────────────────────────

function SendIcon() {
  return (
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M22 2L11 13" />
      <path d="M22 2L15 22 11 13 2 9l20-7z" />
    </svg>
  )
}

function CalendarIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
      <line x1="16" y1="2" x2="16" y2="6" />
      <line x1="8" y1="2" x2="8" y2="6" />
      <line x1="3" y1="10" x2="21" y2="10" />
    </svg>
  )
}

function ChevronIcon({ open }: { open: boolean }) {
  return (
    <svg
      width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"
      className={`transition-transform duration-200 ${open ? 'rotate-180' : ''}`}
    >
      <polyline points="6 9 12 15 18 9" />
    </svg>
  )
}

// ── main component ────────────────────────────────────────────────────────────

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([WELCOME])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [showSuggestions, setShowSuggestions] = useState(true)
  const nextId = useRef(1)
  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Auto-resize textarea
  useEffect(() => {
    const el = inputRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 120) + 'px'
  }, [input])

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

  function onKey(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  const hasContent = input.trim().length > 0

  return (
    <div className="flex flex-col h-screen bg-slate-50">
      {/* ── header ── */}
      <header className="relative z-10 flex items-center gap-3.5 px-5 py-3.5 bg-white border-b border-slate-100 shadow-[0_1px_12px_rgba(0,0,0,0.06)]">
        {/* gradient avatar */}
        <div
          className="w-10 h-10 rounded-xl flex items-center justify-center text-white font-bold text-sm flex-shrink-0"
          style={{ background: 'linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)' }}
        >
          BM
        </div>

        {/* name + status */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <p className="font-semibold text-slate-800 text-sm leading-tight">Bhuvanesh M S</p>
            <span className="inline-flex items-center gap-1 text-[10px] font-medium text-emerald-600 bg-emerald-50 border border-emerald-100 px-1.5 py-0.5 rounded-full">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
              AI Active
            </span>
          </div>
          <p className="text-[11px] text-slate-400 mt-0.5 truncate">
            AI Engineer · Scaler + BITS Pilani
          </p>
        </div>

        {/* book a call */}
        {CALCOM_URL && (
          <a
            href={CALCOM_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="flex-shrink-0 flex items-center gap-1.5 text-xs font-semibold
                       bg-indigo-600 hover:bg-indigo-700 active:bg-indigo-800
                       text-white px-3.5 py-2 rounded-lg transition-colors shadow-sm"
          >
            <CalendarIcon />
            Book a Call
          </a>
        )}
      </header>

      {/* ── messages ── */}
      <main className="flex-1 overflow-y-auto scrollbar-thin">
        <div className="max-w-2xl mx-auto px-4 py-6 space-y-4">

          {messages.map((msg, i) => (
            <div key={msg.id} className="msg-enter" style={{ animationDelay: `${i === messages.length - 1 ? 0 : 0}ms` }}>
              <MessageBubble msg={msg} />
            </div>
          ))}

          {/* suggestion chips */}
          {showSuggestions && !loading && (
            <div className="msg-enter">
              <p className="text-[11px] text-slate-400 mb-2 ml-11">Suggested questions</p>
              <div className="flex flex-wrap gap-2 ml-11">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s.label}
                    onClick={() => send(s.prompt)}
                    className="text-xs px-3 py-1.5 rounded-full border border-indigo-200 text-indigo-600
                               bg-white hover:bg-indigo-50 hover:border-indigo-400 hover:shadow-sm
                               transition-all duration-150 font-medium"
                  >
                    {s.label}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div ref={bottomRef} />
        </div>
      </main>

      {/* ── input bar ── */}
      <footer className="bg-white border-t border-slate-100 px-4 pt-3 pb-4 shadow-[0_-1px_12px_rgba(0,0,0,0.04)]">
        <div className="max-w-2xl mx-auto">
          <div className={`flex items-end gap-2 rounded-2xl border bg-slate-50 px-4 py-2.5 transition-all duration-150
            ${hasContent || loading
              ? 'border-indigo-300 ring-2 ring-indigo-100 bg-white'
              : 'border-slate-200 focus-within:border-indigo-300 focus-within:ring-2 focus-within:ring-indigo-100 focus-within:bg-white'
            }`}
          >
            <textarea
              ref={inputRef}
              rows={1}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKey}
              disabled={loading}
              placeholder="Ask about Bhuvanesh's skills, projects, or experience…"
              className="flex-1 bg-transparent text-sm text-slate-800 placeholder-slate-400
                         outline-none resize-none leading-relaxed disabled:opacity-50
                         disabled:cursor-not-allowed min-h-[24px] max-h-[120px] py-0.5"
            />
            <button
              onClick={() => send()}
              disabled={loading || !hasContent}
              className={`flex-shrink-0 flex items-center justify-center w-8 h-8 rounded-xl
                         transition-all duration-150 mb-0.5
                         ${hasContent && !loading
                           ? 'bg-indigo-600 text-white hover:bg-indigo-700 active:scale-95 shadow-sm'
                           : 'bg-slate-100 text-slate-300 cursor-not-allowed'
                         }`}
            >
              {loading ? (
                <span className="flex items-center gap-0.5">
                  <span className="w-1 h-1 bg-current rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                  <span className="w-1 h-1 bg-current rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                  <span className="w-1 h-1 bg-current rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                </span>
              ) : (
                <SendIcon />
              )}
            </button>
          </div>
          <p className="text-center text-[10px] text-slate-300 mt-2">
            AI-generated · responses may not reflect real-time information · Press Enter to send
          </p>
        </div>
      </footer>
    </div>
  )
}

// ── avatar ────────────────────────────────────────────────────────────────────

function Avatar() {
  return (
    <div
      className="w-8 h-8 rounded-xl flex items-center justify-center text-white text-[10px]
                 font-bold flex-shrink-0 self-end mb-0.5"
      style={{ background: 'linear-gradient(135deg, #6366f1, #8b5cf6)' }}
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

      <div className={`max-w-[78%] flex flex-col gap-1.5 ${isUser ? 'items-end' : 'items-start'}`}>
        {/* bubble */}
        <div
          className={`rounded-2xl px-4 py-3 text-sm leading-relaxed min-h-[40px] ${
            isUser
              ? 'bg-indigo-600 text-white rounded-br-sm whitespace-pre-wrap'
              : 'bg-white text-slate-800 border border-slate-100 shadow-sm rounded-bl-sm'
          }`}
        >
          {isUser ? (
            msg.content
          ) : (
            <>
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                  ul: ({ children }) => <ul className="list-disc pl-4 mb-2 space-y-0.5">{children}</ul>,
                  ol: ({ children }) => <ol className="list-decimal pl-4 mb-2 space-y-0.5">{children}</ol>,
                  li: ({ children }) => <li className="leading-relaxed">{children}</li>,
                  strong: ({ children }) => <strong className="font-semibold text-slate-900">{children}</strong>,
                  em: ({ children }) => <em className="italic text-slate-600">{children}</em>,
                  code: ({ children }) => (
                    <code className="bg-slate-100 text-indigo-600 px-1.5 py-0.5 rounded-md text-xs font-mono border border-slate-200">
                      {children}
                    </code>
                  ),
                  pre: ({ children }) => (
                    <pre className="bg-slate-900 text-slate-100 p-3 rounded-xl text-xs overflow-x-auto mb-2 font-mono leading-relaxed">
                      {children}
                    </pre>
                  ),
                  h1: ({ children }) => <h1 className="font-bold text-base text-slate-900 mb-1.5 mt-1">{children}</h1>,
                  h2: ({ children }) => <h2 className="font-semibold text-sm text-slate-900 mb-1.5 mt-1">{children}</h2>,
                  h3: ({ children }) => <h3 className="font-semibold text-sm text-slate-700 mb-1 mt-1">{children}</h3>,
                  blockquote: ({ children }) => (
                    <blockquote className="border-l-2 border-indigo-300 pl-3 my-2 text-slate-500 italic text-sm">
                      {children}
                    </blockquote>
                  ),
                  a: ({ href, children }) => (
                    <a href={href} target="_blank" rel="noopener noreferrer"
                       className="text-indigo-600 hover:text-indigo-800 underline underline-offset-2">
                      {children}
                    </a>
                  ),
                  hr: () => <hr className="border-slate-100 my-2" />,
                }}
              >
                {msg.content}
              </ReactMarkdown>

              {/* typing dots */}
              {msg.streaming && !msg.content && (
                <span className="inline-flex items-center gap-1 h-4">
                  <span className="w-2 h-2 bg-slate-300 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                  <span className="w-2 h-2 bg-slate-300 rounded-full animate-bounce" style={{ animationDelay: '160ms' }} />
                  <span className="w-2 h-2 bg-slate-300 rounded-full animate-bounce" style={{ animationDelay: '320ms' }} />
                </span>
              )}

              {/* blinking cursor */}
              {msg.streaming && msg.content && (
                <span className="ml-0.5 inline-block w-0.5 h-3.5 bg-indigo-400 animate-pulse align-middle rounded-full" />
              )}
            </>
          )}
        </div>

        {/* sources */}
        {msg.sources && msg.sources.length > 0 && (
          <div className="w-full px-1">
            <button
              onClick={() => setOpen((v) => !v)}
              className="flex items-center gap-1 text-[11px] text-slate-400 hover:text-indigo-500
                         transition-colors font-medium"
            >
              <ChevronIcon open={open} />
              {open ? 'Hide sources' : `${msg.sources.length} source${msg.sources.length > 1 ? 's' : ''}`}
            </button>
            {open && (
              <ul className="mt-1.5 pl-2 border-l-2 border-indigo-100 space-y-1">
                {msg.sources.map((s) => (
                  <li key={s.n} className="flex items-center gap-1.5 text-[11px] text-slate-400">
                    <span className="font-mono text-slate-300 shrink-0">[{s.n}]</span>
                    <span className="truncate">{s.title}</span>
                    {s.is_fork && (
                      <span className="shrink-0 text-[9px] border border-amber-200 text-amber-500 bg-amber-50 px-1.5 py-0.5 rounded-full">
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
