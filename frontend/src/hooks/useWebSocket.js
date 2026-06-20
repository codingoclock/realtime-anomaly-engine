import { useEffect, useRef, useCallback } from 'react'

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws/anomalies'
const RECONNECT_DELAY_MS = 2500

/**
 * useWebSocket(onMessage)
 *
 * Opens a WebSocket to the Anomaly API and calls `onMessage` with each parsed
 * JSON event. Automatically reconnects on close/error with a fixed backoff.
 *
 * Returns a `disconnect` callback to close the socket manually.
 */
export function useWebSocket(onMessage) {
  const wsRef = useRef(null)
  const timerRef = useRef(null)
  const mountedRef = useRef(true)

  const connect = useCallback(() => {
    if (!mountedRef.current) return

    const ws = new WebSocket(WS_URL)
    wsRef.current = ws

    ws.onopen = () => {
      console.log('[WS] Connected to', WS_URL)
      // Send an initial keepalive so the server detects us
      ws.send('ping')
    }

    ws.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data)
        onMessage(data)
      } catch (err) {
        console.warn('[WS] Failed to parse message:', err)
      }
    }

    ws.onclose = () => {
      console.log('[WS] Disconnected — reconnecting in', RECONNECT_DELAY_MS, 'ms')
      if (mountedRef.current) {
        timerRef.current = setTimeout(connect, RECONNECT_DELAY_MS)
      }
    }

    ws.onerror = (err) => {
      console.warn('[WS] Error:', err)
      ws.close()
    }
  }, [onMessage])

  useEffect(() => {
    mountedRef.current = true
    connect()
    return () => {
      mountedRef.current = false
      clearTimeout(timerRef.current)
      if (wsRef.current) {
        wsRef.current.onclose = null // prevent reconnect loop on unmount
        wsRef.current.close()
      }
    }
  }, [connect])

  const disconnect = useCallback(() => {
    mountedRef.current = false
    clearTimeout(timerRef.current)
    if (wsRef.current) wsRef.current.close()
  }, [])

  return { disconnect }
}
