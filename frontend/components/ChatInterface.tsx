'use client'

import { useState, useRef, useEffect, KeyboardEvent } from 'react'

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

// ── main component ────────────────────────────────────────────────────────────

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([WELCOME])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const nextId = useRef(1)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  function buildHistory() {
    return messages
      .filter((m) => !m.streaming && m.id !== 0)
      .slice(-12)
      .map((m) => ({ role: m.role, content: m.content }))
  }

  async function send() {
    const q = input.trim()
    if (!q || loading) return
    setInput('')
    setLoading(true)

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

        // SSE blocks are separated by \n\n
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
            ? { ...m, content: 'Something went wrong — please try again.', streaming: false }
            : m
        )
      )
    } finally {
      setMessages((prev) =>
        prev.map((m) => (m.id === assistantId ? { ...m, streaming: false } : m))
      )
      setLoading(false)
    }
  }

  function onKey(e: KeyboardEvent<HTMLInputElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      send()
    }
  }

  return (
    <div className="flex flex-col h-screen bg-slate-50">
      {/* ── header ── */}
      <header className="bg-white border-b border-slate-200 px-4 py-3 flex items-center gap-3 shadow-sm">
        <Avatar size="md" />
        <div className="flex-1 min-w-0">
          <p className="font-semibold text-slate-900 text-sm leading-none">Bhuvanesh M S</p>
          <p className="text-xs text-slate-500 mt-0.5 truncate">
            AI Engineer · Scaler School of Technology · BITS Pilani
          </p>
        </div>
        {CALCOM_URL && (
          <a
            href={CALCOM_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="flex-shrink-0 text-xs font-medium bg-indigo-600 hover:bg-indigo-700 text-white px-3 py-1.5 rounded-full transition-colors"
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
          <div ref={bottomRef} />
        </div>
      </main>

      {/* ── input ── */}
      <footer className="bg-white border-t border-slate-200 px-4 py-3">
        <div className="max-w-2xl mx-auto flex gap-2">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKey}
            disabled={loading}
            placeholder="Ask about Bhuvanesh's skills, projects, or experience…"
            className="flex-1 rounded-xl border border-slate-300 px-4 py-2.5 text-sm outline-none
                       focus:ring-2 focus:ring-indigo-500 focus:border-transparent
                       disabled:bg-slate-50 disabled:text-slate-400"
          />
          <button
            onClick={send}
            disabled={loading || !input.trim()}
            className="rounded-xl bg-indigo-600 px-4 py-2.5 text-sm font-medium text-white
                       transition hover:bg-indigo-700 disabled:bg-slate-300 disabled:cursor-not-allowed
                       flex-shrink-0"
          >
            {loading ? '…' : 'Send'}
          </button>
        </div>
      </footer>
    </div>
  )
}

// ── sub-components ────────────────────────────────────────────────────────────

function Avatar({ size }: { size: 'sm' | 'md' }) {
  const cls = size === 'sm' ? 'w-7 h-7 text-xs' : 'w-10 h-10 text-sm'
  return (
    <div
      className={`${cls} rounded-full bg-indigo-600 flex items-center justify-center
                  text-white font-semibold flex-shrink-0`}
    >
      BM
    </div>
  )
}

function MessageBubble({ msg }: { msg: Message }) {
  const [open, setOpen] = useState(false)
  const isUser = msg.role === 'user'

  return (
    <div className={`flex gap-2.5 ${isUser ? 'flex-row-reverse' : ''}`}>
      {!isUser && <Avatar size="sm" />}

      <div className={`max-w-[78%] flex flex-col ${isUser ? 'items-end' : 'items-start'} gap-1.5`}>
        {/* bubble */}
        <div
          className={`rounded-2xl px-4 py-2.5 text-sm leading-relaxed whitespace-pre-wrap ${
            isUser
              ? 'bg-indigo-600 text-white rounded-tr-none'
              : 'bg-white text-slate-800 border border-slate-200 shadow-sm rounded-tl-none'
          }`}
        >
          {msg.content}

          {/* typing indicator — empty streaming message */}
          {msg.streaming && !msg.content && (
            <span className="inline-flex gap-1 items-center">
              <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce [animation-delay:0ms]" />
              <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce [animation-delay:150ms]" />
              <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce [animation-delay:300ms]" />
            </span>
          )}

          {/* blinking cursor while streaming text */}
          {msg.streaming && msg.content && (
            <span className="ml-0.5 inline-block w-0.5 h-3.5 bg-slate-500 animate-pulse align-middle" />
          )}
        </div>

        {/* sources */}
        {msg.sources && msg.sources.length > 0 && (
          <div>
            <button
              onClick={() => setOpen((v) => !v)}
              className="text-[11px] text-slate-400 hover:text-slate-600 transition-colors"
            >
              {open
                ? '▾ hide sources'
                : `▸ ${msg.sources.length} source${msg.sources.length > 1 ? 's' : ''}`}
            </button>
            {open && (
              <ul className="mt-1 pl-2 border-l border-slate-200 space-y-0.5">
                {msg.sources.map((s) => (
                  <li key={s.n} className="flex items-center gap-1.5 text-[11px] text-slate-400">
                    <span className="font-mono text-slate-300">[{s.n}]</span>
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
