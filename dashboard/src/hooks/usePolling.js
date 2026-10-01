import { useEffect, useEffectEvent } from "react";

/**
 * Run `callback` now and every `intervalMs` while `enabled`.
 *
 * `useEffectEvent` always sees the latest props/state, so the interval is not
 * torn down on every render and never calls a stale closure (the v2 dashboard
 * captured `latest`/`history` once and silently dropped updates).
 */
export function usePolling(callback, intervalMs, enabled = true) {
  const onTick = useEffectEvent(callback);

  useEffect(() => {
    if (!enabled) return undefined;
    onTick();
    const id = setInterval(() => onTick(), intervalMs);
    return () => clearInterval(id);
  }, [intervalMs, enabled]);
}
