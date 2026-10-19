import { useQuery } from "@tanstack/react-query";

import { api } from "../api";
import { FindingCard, VerdictCard } from "../components/Findings";
import { Bone, EmptyState, Failure, Loading, Notice, PageHeader } from "../components/ui";

export function FindingsPage() {
  const claims = useQuery({ queryKey: ["claims"], queryFn: api.claims });
  const data = claims.data;
  const count = data ? data.claims.length + data.verdicts.length : null;
  return (
    <>
      <PageHeader
        title={count === null ? "Findings" : `Findings (${count} claims)`}
        lead="The allow-listed claims of the evidence ledger: each value with its interval and the clients behind it, and each headline with the robustness checker's verdict and the reason for it."
      />
      {claims.error ? <Failure error={claims.error} onRetry={() => void claims.refetch()} /> : null}
      {claims.isPending ? (
        <Loading what="the claims">
          <div className="grid grid-cols-1 gap-5 xl:grid-cols-2">
            {[0, 1, 2, 3].map((index) => (
              <div key={index} className="flex flex-col gap-3 rounded-panel border border-border bg-surface p-5">
                <Bone className="h-3 w-40" />
                <Bone className="h-5 w-full" />
                <Bone className="h-4 w-3/4" />
                <Bone className="h-24 w-full" />
              </div>
            ))}
          </div>
        </Loading>
      ) : null}
      {data ? (
        <div className="flex flex-col gap-5">
          <Notice kind="status">
            {data.source_kind === "real"
              ? "These claims are computed on the real reference extract and published as allow-listed aggregates; every other page shows synthetic data."
              : "These claims are computed on synthetic data, like every other page."}
          </Notice>
          {count === 0 ? (
            <EmptyState title="No claims are published yet." command="just publish-claims">
              Claims appear here once the private zone publishes its allow-listed aggregates to the public ledger.
            </EmptyState>
          ) : (
            <div className="grid grid-cols-1 items-start gap-5 xl:grid-cols-2">
              {data.claims.map((claim) => (
                <FindingCard key={claim.id} claim={claim} />
              ))}
              {data.verdicts.map((verdict) => (
                <VerdictCard key={verdict.id} verdict={verdict} />
              ))}
            </div>
          )}
        </div>
      ) : null}
    </>
  );
}
