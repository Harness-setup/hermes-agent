import { act, renderHook } from '@testing-library/react'
import { type ReactNode } from 'react'
import { beforeEach, expect, it, vi } from 'vitest'

import { PaneVisibleContext } from '@/components/pane-shell/pane-visibility'
import { $gateway } from '@/store/gateway'

import { useInputShown } from './use-input-shown'

const route = vi.hoisted(() => vi.fn(async () => ({ acknowledged: true })))
vi.mock('@/store/session-states', () => ({ requestForOwnedSession: route }))

beforeEach(() => {
  route.mockClear()
  $gateway.set({ request: vi.fn() } as unknown as NonNullable<ReturnType<typeof $gateway.get>>)
})

it('acknowledges the owning session and request after an actionable panel mounts', () => {
  const { rerender } = renderHook(({ requestId }) => useInputShown('owner', requestId), {
    initialProps: { requestId: undefined as string | undefined }
  })

  expect(route).not.toHaveBeenCalled()
  rerender({ requestId: 'srq-question' })
  expect(route).toHaveBeenCalledWith('owner', expect.any(Function), 'request.shown', {
    request_id: 'srq-question',
    session_id: 'owner'
  })
})

it('acknowledges a kept pane only once it becomes visible', () => {
  let visible = false

  const wrapper = ({ children }: { children: ReactNode }) => (
    <PaneVisibleContext.Provider value={visible}>{children}</PaneVisibleContext.Provider>
  )

  const { rerender } = renderHook(() => useInputShown('owner', 'srq-hidden'), { wrapper })
  expect(route).not.toHaveBeenCalled()
  visible = true
  rerender()
  expect(route).toHaveBeenCalledTimes(1)
})


it('acknowledges a mounted prompt when its owning connection becomes available', () => {
  $gateway.set(null)
  renderHook(() => useInputShown('owner', 'srq-reconnect'))
  expect(route).not.toHaveBeenCalled()
  act(() => {
    $gateway.set({ request: vi.fn() } as unknown as NonNullable<ReturnType<typeof $gateway.get>>)
  })
  expect(route).toHaveBeenCalledTimes(1)
})
