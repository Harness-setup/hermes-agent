import { describe, expect, it, vi } from 'vitest'

import {
  applyConnectionChange,
  commitConnectionFailure,
  resolveTerminalConnection,
  sshQuitShouldBlock,
  teardownSshState
} from './connection-apply'

function deferred() {
  let resolve!: () => void

  const promise = new Promise<void>(done => {
    resolve = done
  })

  return { promise, resolve }
}

describe('applyConnectionChange', () => {
  it.each([['SSH A to SSH B'], ['SSH to Cloud'], ['Cloud to SSH']])(
    'serializes %s behind bootstrap rollback before teardown and apply',
    async () => {
      const gate = deferred()
      const events: string[] = []

      const run = applyConnectionChange({
        cancelAndWait: async () => {
          events.push('cancel')
          await gate.promise
          events.push('drained')
        },
        isPrimary: true,
        scope: '',
        sendApplied: () => events.push('applied'),
        stopPool: vi.fn(),
        teardownPrimary: async () => {
          events.push('primary')
        },
        teardownSsh: async () => {
          events.push('ssh')
        }
      })

      await Promise.resolve()
      expect(events).toEqual(['cancel'])
      gate.resolve()
      await run
      expect(events).toEqual(['cancel', 'drained', 'ssh', 'primary', 'applied'])
    }
  )

  it('tears down only a non-primary scope without applying the primary connection', async () => {
    const events: string[] = []
    await applyConnectionChange({
      cancelAndWait: async scope => {
        events.push(`cancel:${scope}`)
      },
      isPrimary: false,
      scope: 'worker',
      sendApplied: () => events.push('applied'),
      stopPool: scope => events.push(`pool:${scope}`),
      teardownPrimary: async () => {
        events.push('primary')
      },
      teardownSsh: async scope => {
        events.push(`ssh:${scope}`)
      }
    })
    expect(events).toEqual(['cancel:worker', 'ssh:worker', 'pool:worker'])
  })
})

describe('resolveTerminalConnection', () => {
  it('joins an in-flight backend before resolving the SSH terminal target', async () => {
    const target = { ssh: {}, scope: '' }
    const getTarget = vi.fn().mockReturnValueOnce('pending').mockReturnValueOnce(target)
    const ensureBackend = vi.fn(async () => undefined)

    await expect(resolveTerminalConnection(getTarget, ensureBackend)).resolves.toBe(target)
    expect(ensureBackend).toHaveBeenCalledOnce()
  })

  it('does not start a local terminal while configured SSH remains unavailable', async () => {
    const ensureBackend = vi.fn(async () => undefined)

    await expect(
      resolveTerminalConnection(
        () => 'pending',
        ensureBackend,
        async () => undefined // no real delay -- getTarget stays 'pending' regardless
      )
    ).rejects.toThrow('not ready')
    // A persistently-unavailable target must not retry the backend itself
    // over and over -- only re-poll the already-cheap getTarget() call.
    expect(ensureBackend).toHaveBeenCalledOnce()
  })

  it('resolves once getTarget flips after a few polls, without calling ensureBackend again', async () => {
    // Tony: "Terminal failed to start: ... Remote connection is not ready
    // yet." -- the real bug this covers: getTarget() can still say
    // 'pending' immediately after ensureBackend()'s single await
    // resolves (a real SSH/WSL cold start finishing a moment later, not
    // synchronized with that exact promise). Simulates getTarget flipping
    // to the real target on the 4th poll, not the 1st.
    const target = { ssh: {}, scope: '' }
    const getTarget = vi
      .fn()
      .mockReturnValueOnce('pending') // pre-check
      .mockReturnValueOnce('pending') // immediately after ensureBackend
      .mockReturnValueOnce('pending') // poll 1
      .mockReturnValueOnce('pending') // poll 2
      .mockReturnValueOnce('pending') // poll 3
      .mockReturnValueOnce(target) // poll 4
    const ensureBackend = vi.fn(async () => undefined)
    const delays: number[] = []

    await expect(
      resolveTerminalConnection(getTarget, ensureBackend, async ms => {
        delays.push(ms)
      })
    ).resolves.toBe(target)
    expect(ensureBackend).toHaveBeenCalledOnce()
    expect(delays).toHaveLength(4)
  })
})

describe('sshQuitShouldBlock', () => {
  it('waits when connections exist and teardown has not finished', () => {
    expect(sshQuitShouldBlock({ teardownDone: false, connectionCount: 1, bootstrapPending: 0, inFlight: null })).toBe(
      true
    )
  })

  it('waits when bootstrap is still running', () => {
    expect(sshQuitShouldBlock({ teardownDone: false, connectionCount: 0, bootstrapPending: 1, inFlight: null })).toBe(
      true
    )
  })

  it('waits when the map is empty but a remote kill is already in flight', () => {
    expect(
      sshQuitShouldBlock({
        teardownDone: false,
        connectionCount: 0,
        bootstrapPending: 0,
        inFlight: Promise.resolve()
      })
    ).toBe(true)
  })

  it('does not block a second quit after teardown finished', () => {
    expect(
      sshQuitShouldBlock({
        teardownDone: true,
        connectionCount: 1,
        bootstrapPending: 1,
        inFlight: Promise.resolve()
      })
    ).toBe(false)
  })

  it('does not block quit when there is nothing to tear down', () => {
    expect(sshQuitShouldBlock({ teardownDone: false, connectionCount: 0, bootstrapPending: 0, inFlight: null })).toBe(
      false
    )
  })
})

describe('teardownSshState', () => {
  it('terminates the owned remote backend before closing its tunnel and SSH transport', async () => {
    const events: string[] = []

    const ssh = {
      cancelForward: async () => events.push('forward'),
      close: async () => events.push('ssh')
    }

    await teardownSshState(
      { ssh, ownershipId: 'owner', localPort: 1234, remotePort: 5678 },
      { cleanupRemote: async () => events.push('remote') }
    )

    expect(events).toEqual(['remote', 'forward', 'ssh'])
  })

  it('still closes the SSH transport when remote cleanup fails', async () => {
    const close = vi.fn(async () => undefined)

    await teardownSshState(
      { ssh: { cancelForward: vi.fn(async () => undefined), close }, ownershipId: 'owner' },
      {
        cleanupRemote: async () => {
          throw new Error('remote unavailable')
        }
      }
    )

    expect(close).toHaveBeenCalledOnce()
  })
})

describe('commitConnectionFailure', () => {
  it('prevents a stale bootstrap from publishing failure state', () => {
    const stale = Promise.resolve('stale')
    const current = Promise.resolve('current')
    const commit = vi.fn()

    expect(commitConnectionFailure(current, stale, commit)).toBe(false)
    expect(commit).not.toHaveBeenCalled()
    expect(commitConnectionFailure(current, current, commit)).toBe(true)
    expect(commit).toHaveBeenCalledOnce()
  })
})
