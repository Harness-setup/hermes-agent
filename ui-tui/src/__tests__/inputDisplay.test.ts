import { expect, it, vi } from 'vitest'

import {
  displayServerRequest,
  forgetServerRequest,
  rememberServerRequest,
  resetServerRequestsForTests
} from '../app/serverRequestStore.js'

it('acknowledges the rendered request using its owning session, and ignores finished requests', () => {
  resetServerRequestsForTests()
  const rpc = vi.fn(async () => ({}))
  rememberServerRequest({
    id: 'srq-question',
    method: 'sudo',
    params: { session_id: 'owner' },
    respond: vi.fn(),
    fail: vi.fn()
  })
  displayServerRequest('srq-question', rpc)
  expect(rpc).toHaveBeenCalledWith('request.shown', { request_id: 'srq-question', session_id: 'owner' })
  forgetServerRequest('srq-question')
  displayServerRequest('srq-question', rpc)
  expect(rpc).toHaveBeenCalledTimes(1)
})
