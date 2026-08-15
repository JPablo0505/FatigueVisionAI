import { useEffect, useRef, useState } from 'react'

export type CameraPermission = 'granted' | 'denied' | 'pending'

export interface CameraStream {
  stream: MediaStream | null
  error: string | null
  permission: CameraPermission
}

const CAMERA_CONSTRAINTS: MediaStreamConstraints = {
  video: { width: 640, height: 480, facingMode: 'user' },
  audio: false,
}

/** Traduce un error de getUserMedia a un mensaje claro para el usuario. */
function cameraErrorMessage(err: unknown): string {
  const name = err instanceof DOMException || err instanceof Error ? err.name : ''
  switch (name) {
    case 'NotAllowedError':
    case 'PermissionDeniedError':
      return 'Camera permission denied. Allow camera access to see the live feed.'
    case 'NotFoundError':
    case 'DevicesNotFoundError':
      return 'No camera found. Connect a webcam and reload the dashboard.'
    case 'NotReadableError':
      return 'Camera is in use by another application.'
    case 'NotSupportedError':
    case 'OverconstrainedError':
    case 'TypeError':
      return 'Camera is not supported or the requested resolution is unavailable.'
    default:
      return 'Unable to access the camera.'
  }
}

/**
 * Expone la cámara local (getUserMedia 640x480). No procesa nada de video:
 * solo adquiere el stream, lo devuelve y detiene los tracks al desmontar.
 */
export function useCameraStream(): CameraStream {
  const [stream, setStream] = useState<MediaStream | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [permission, setPermission] = useState<CameraPermission>('pending')

  const streamRef = useRef<MediaStream | null>(null)

  useEffect(() => {
    let disposed = false

    const stopStream = (target: MediaStream | null) => {
      if (target !== null) {
        for (const track of target.getTracks()) {
          track.stop()
        }
      }
    }

    async function start() {
      const media = navigator.mediaDevices
      if (media === undefined) {
        setPermission('denied')
        setError('Camera access is not supported in this browser.')
        return
      }
      try {
        const acquired = await media.getUserMedia(CAMERA_CONSTRAINTS)
        if (disposed) {
          stopStream(acquired)
          return
        }
        streamRef.current = acquired
        setStream(acquired)
        setPermission('granted')
        setError(null)
      } catch (err) {
        if (disposed) return
        streamRef.current = null
        setStream(null)
        setPermission('denied')
        setError(cameraErrorMessage(err))
      }
    }

    void start()

    return () => {
      disposed = true
      stopStream(streamRef.current)
      streamRef.current = null
      setStream(null)
    }
  }, [])

  return { stream, error, permission }
}