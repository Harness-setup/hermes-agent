import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { usePaneVisible } from '@/components/pane-shell/pane-visibility'
import { $gateway } from '@/store/gateway'
import { ambientRequestFor } from '@/store/session-gone-latch'
import { requestForOwnedSession } from '@/store/session-states'

/** Call from the mounted, actionable prompt, after React commits its panel. */
export function useInputShown(sessionId: string | null | undefined, requestId: string | undefined): void {
  const visible = usePaneVisible()
  const gateway = useStore($gateway)
  useEffect(() => {
    if (!visible || !gateway || !sessionId || !requestId) {return}
    void requestForOwnedSession(sessionId, ambientRequestFor(gateway), 'request.shown', {
      session_id: sessionId,
      request_id: requestId
    }).catch(() => {
      /* Older backends do not support display acknowledgments. */
    })
  }, [visible, gateway, sessionId, requestId])
}

interface ApprovalDisplayRequest {
  sessionId: string | null
  serverRequestId?: string
  requestId?: string
}

export function useApprovalInputShown(request: ApprovalDisplayRequest, active: boolean): void {
  useInputShown(request.sessionId, active ? (request.serverRequestId ?? request.requestId) : undefined)
}
