import { useQuery } from "@tanstack/react-query";

import { api } from "../api";
import { FindingCard, VerdictCard } from "../components/Findings";
import { Failure, Loading, PageHeader } from "../components/ui";

export function FindingsPage() {
  const claims = useQuery({ queryKey: ["claims"], queryFn: api.claims });
  return (
    <>
      <PageHeader
        title="Findings"
        lead="The allow-listed claims of the evidence ledger, computed on the reference instance: each value with its interval and the clients behind it, and each headline with the robustness checker's verdict."
      />
      {claims.isPending ? <Loading what="the claims" /> : null}
      {claims.error ? <Failure error={claims.error} /> : null}
      {claims.data ? (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          {claims.data.claims.map((claim) => (
            <FindingCard key={claim.id} claim={claim} />
          ))}
          {claims.data.verdicts.map((verdict) => (
            <VerdictCard key={verdict.id} verdict={verdict} />
          ))}
        </div>
      ) : null}
    </>
  );
}
