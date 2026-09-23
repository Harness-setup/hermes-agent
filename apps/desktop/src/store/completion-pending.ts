/**
 * Mirror of "is any composer waiting on a slash/@ completion RPC" to the main
 * process, so it can keep this window's painting unthrottled the same way a
 * live turn does (electron/stream-throttle.ts).
 *
 * Deliberately separate from store/active-work.ts, not a reuse of it: that
 * store's ActiveWork payload also drives the quit-confirmation dialog ("a
 * turn is running, quit anyway?"), and a pending completion is not unsaved
 * work -- it must never make quitting look destructive.
 *
 * Live-diagnosed 2026-08-17: a slash command's arg-stage dropdown (e.g.
 * `/mode `) could fetch correctly and update React state correctly, but the
 * repaint that shows it landed inside Chromium's default background
 * throttling window and got silently deferred until an unrelated event (a
 * focus change, a keystroke) forced a paint -- indistinguishable from the
 * completion "just not working" without a debugger attached. Only a live
 * turn was previously exempted from that throttling; this exempts a pending
 * completion the same way.
 *
 * Multiple composer instances (main window, HUD, popped-out tiles) can each
 * have their own pending completion at once, so this is a reference-counted
 * registry keyed by caller-supplied instance id, not a single boolean.
 */

const pendingIds = new Set<string>()
let lastSent: boolean | null = null

function publish(): void {
  const pending = pendingIds.size > 0

  if (pending === lastSent) {
    return
  }

  lastSent = pending

  if (typeof window !== 'undefined') {
    window.hermesDesktop?.setCompletionPending?.(pending)
  }
}

/** Register or clear one composer instance's pending-completion state.
 *  Call with `pending: false` (or `unregisterCompletionPending`) on
 *  unmount so a closed tile never leaves a stuck "pending" vote behind. */
export function setCompletionPending(id: string, pending: boolean): void {
  if (pending) {
    pendingIds.add(id)
  } else {
    pendingIds.delete(id)
  }

  publish()
}

export function unregisterCompletionPending(id: string): void {
  pendingIds.delete(id)
  publish()
}
