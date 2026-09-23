import { beforeEach, describe, expect, it, vi } from 'vitest'

import { setCompletionPending, unregisterCompletionPending } from './completion-pending'

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const setCompletionPendingBridge = vi.fn()

beforeEach(() => {
  desktopWindow.hermesDesktop = { setCompletionPending: setCompletionPendingBridge } as unknown as Window['hermesDesktop']
  setCompletionPendingBridge.mockClear()
  // Drain any registrations a previous test left behind -- the module-level
  // Set and `lastSent` dedup latch persist across tests in the same file.
  unregisterCompletionPending('a')
  unregisterCompletionPending('b')
  setCompletionPendingBridge.mockClear()
})

describe('completion-pending bridge', () => {
  it('reports true when a composer starts a pending completion', () => {
    setCompletionPending('a', true)

    expect(setCompletionPendingBridge).toHaveBeenLastCalledWith(true)
  })

  it('reports false once the only pending composer clears', () => {
    setCompletionPending('a', true)
    setCompletionPendingBridge.mockClear()

    setCompletionPending('a', false)

    expect(setCompletionPendingBridge).toHaveBeenLastCalledWith(false)
  })

  it('stays true while a second composer is still pending', () => {
    setCompletionPending('a', true)
    setCompletionPending('b', true)
    setCompletionPendingBridge.mockClear()

    setCompletionPending('a', false)

    expect(setCompletionPendingBridge).not.toHaveBeenCalled()
  })

  it('does not re-send an unchanged value', () => {
    setCompletionPending('a', true)
    setCompletionPendingBridge.mockClear()

    setCompletionPending('b', true)

    expect(setCompletionPendingBridge).not.toHaveBeenCalled()
  })

  it('unregister clears a stuck vote on unmount, matching a false report', () => {
    setCompletionPending('a', true)
    setCompletionPendingBridge.mockClear()

    unregisterCompletionPending('a')

    expect(setCompletionPendingBridge).toHaveBeenLastCalledWith(false)
  })
})
