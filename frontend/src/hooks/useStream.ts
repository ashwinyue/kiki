/**
 * Kiki Agent Framework - SSE 流式 Hook
 */

import { useCallback, useEffect, useRef } from 'react';
import type { StreamEvent } from '@/types/chat';

export interface UseStreamOptions {
  onToken?: (token: string) => void;
  onUpdate?: (data: Record<string, unknown>) => void;
  onState?: (state: Record<string, unknown>) => void;
  onDone?: () => void;
  onError?: (error: Error) => void;
}

/**
 * SSE 流式 Hook
 */
export function useStream(options: UseStreamOptions = {}) {
  const abortControllerRef = useRef<AbortController | null>(null);
  const isStreamingRef = useRef(false);

  const { onToken, onUpdate, onState, onDone, onError } = options;

  // 开始流式传输
  const startStream = useCallback(
    async (url: string, body: Record<string, unknown>) => {
      // 取消之前的请求
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }

      // 创建新的 AbortController
      abortControllerRef.current = new AbortController();
      isStreamingRef.current = true;

      try {
        const response = await fetch(url, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify(body),
          signal: abortControllerRef.current.signal,
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const reader = response.body?.getReader();
        const decoder = new TextDecoder();

        if (!reader) {
          throw new Error('Response body is null');
        }

        // 读取流
        while (true) {
          const { done, value } = await reader.read();

          if (done) {
            isStreamingRef.current = false;
            onDone?.();
            break;
          }

          const chunk = decoder.decode(value, { stream: true });
          const lines = chunk.split('\n');

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = line.slice(6).trim();

              if (!data) continue;

              try {
                const event: StreamEvent = JSON.parse(data);

                switch (event.event) {
                  case 'token':
                    onToken?.(event.data.content || '');
                    break;
                  case 'update':
                    onUpdate?.(event.data);
                    break;
                  case 'state':
                    onState?.(event.data);
                    break;
                  case 'done':
                    isStreamingRef.current = false;
                    onDone?.();
                    break;
                  case 'error':
                    isStreamingRef.current = false;
                    onError?.(new Error(event.data.metadata?.error as string || 'Stream error'));
                    break;
                }
              } catch (parseError) {
                // 忽略解析错误
              }
            }
          }
        }
      } catch (error) {
        isStreamingRef.current = false;

        // 如果是主动取消，不报错
        if (error instanceof Error && error.name === 'AbortError') {
          return;
        }

        onError?.(error as Error);
      }
    },
    [onToken, onUpdate, onState, onDone, onError]
  );

  // 停止流式传输
  const stopStream = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    isStreamingRef.current = false;
  }, []);

  // 组件卸载时取消请求
  useEffect(() => {
    return () => {
      stopStream();
    };
  }, [stopStream]);

  return {
    startStream,
    stopStream,
    isStreaming: () => isStreamingRef.current,
  };
}

export default useStream;
