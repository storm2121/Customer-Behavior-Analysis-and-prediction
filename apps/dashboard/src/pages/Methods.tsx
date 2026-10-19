import { Equation } from "../components/Equation";
import { PageHeader } from "../components/ui";

interface Step {
  prose: string;
  label: string;
  tex: string;
}

const SECTIONS: { title: string; intro: string; steps: Step[]; outro?: string }[] = [
  {
    title: "The robustness checker",
    intro:
      "A headline compares a metric between two groups. The checker recomputes the difference inside the strata of each stratifier and combines the strata into one adjusted difference.",
    steps: [
      {
        prose: "The adjusted difference averages the stratum differences with Mantel-Haenszel weights:",
        label: "Mantel-Haenszel adjusted difference",
        tex: String.raw`\Delta_{\text{adj}} = \frac{\sum_k w_k \Delta_k}{\sum_k w_k}, \qquad w_k = \frac{n_{T,k}\, n_{R,k}}{n_{T,k} + n_{R,k}}`,
      },
    ],
    outro:
      "A stratum's weight grows with the size of both groups in it, so a stratum where one group is thin counts for little. The verdict reverses when the adjusted difference changes sign.",
  },
  {
    title: "Calendar effects",
    intro:
      "Daily counts follow a negative binomial with a log link on the day of the week, bins of the day of the month, sum-coded months and calendar periods; intervals come from a client bootstrap.",
    steps: [
      {
        prose: "The mean of each day's count, and its variance:",
        label: "Negative-binomial calendar model",
        tex: String.raw`\begin{aligned} \log \mu_t &= \beta_0 + \sum_d \gamma_d [\text{dow}_t = d] + \sum_b \delta_b [\text{bin}_t = b] \\ &\quad + \sum_m \theta_m c_{m,t} + \sum_p \lambda_p [t \in p], \\ \operatorname{Var}(Y_t) &= \mu_t + \alpha \mu_t^2 \end{aligned}`,
      },
    ],
    outro: "The dispersion term lets the variance grow faster than the mean, as daily counts of banking events do.",
  },
  {
    title: "The Markov chain and the simulator",
    intro:
      "Each client is in one channel state a month. Every derived quantity is computed on draws from the posterior of the transition matrix, and a policy is projected with the same draws as its baseline.",
    steps: [
      {
        prose: "Each row of the transition matrix has a Dirichlet posterior over the counted transitions:",
        label: "Dirichlet posterior",
        tex: String.raw`P_{i\cdot} \mid N \sim \operatorname{Dirichlet}(1 + N_{i1}, \dots, 1 + N_{iK})`,
      },
      {
        prose: "How long, on average, a client takes to reach one state from another, from the fundamental matrix:",
        label: "Mean first-passage times",
        tex: String.raw`Z = (I - P + \mathbf{1}\pi)^{-1}, \qquad m_{ij} = \frac{z_{jj} - z_{ij}}{\pi_j}`,
      },
      {
        prose: "A policy changes the starting mix; each draw then projects it month by month, and the branch load weights the mix by each state's branch rate:",
        label: "Projection",
        tex: String.raw`x_{h+1} = x_h P^{(s)}, \qquad L_h = x_h \cdot \rho`,
      },
    ],
    outro:
      "Because the projection is linear in the starting mix, every draw's outcome is linear in the share of clients a policy moves; the simulator uses that to redraw its bands while a slider moves.",
  },
  {
    title: "Amounts: empirical Bayes",
    intro: "Unusual amounts are judged against what is typical for the client, borrowing strength from the client's category when the client has few amounts.",
    steps: [
      {
        prose: "The client's typical log amount, shrunk toward the category's:",
        label: "Shrunk mean",
        tex: String.raw`\hat m_i = \frac{t^2 m_i + (s^2/n_i)\,\mu}{t^2 + s^2/n_i}`,
      },
      {
        prose: "A robust z of each amount against that typical value flags the unusual ones:",
        label: "Robust z",
        tex: String.raw`z = \frac{\log a - \hat m_i}{1.4826 \cdot \operatorname{MAD}}`,
      },
    ],
  },
  {
    title: "Leading digits",
    intro: "Amounts that arise freely spread their leading digits in a known way; prices set by hand do not, which is why deviations are read before anything is concluded from them.",
    steps: [
      {
        prose: "Benford's law gives the expected share of each leading digit:",
        label: "Benford's law",
        tex: String.raw`P(D_1 = d) = \log_{10}\!\left(1 + \tfrac{1}{d}\right)`,
      },
      {
        prose: "The mean absolute deviation compares the observed shares with the expected ones, and Nigrini's conformity bands read it:",
        label: "Mean absolute deviation",
        tex: String.raw`\text{MAD} = \frac{1}{K}\sum_k \lvert \hat p_k - p_k \rvert`,
      },
    ],
  },
];

export function MethodsPage() {
  return (
    <>
      <PageHeader
        title="Methods"
        lead="The equations behind the pages, with the reasoning between them. The full derivations, assumptions and limitations are in docs/methods in the repository."
      />
      <article className="flex max-w-measure flex-col gap-10">
        {SECTIONS.map((section, index) => (
          <section key={section.title} aria-labelledby={`method-${index}`} className="flex flex-col gap-3">
            <h2 id={`method-${index}`} className="text-heading font-title text-ink">
              {section.title}
            </h2>
            <p className="text-body text-ink">{section.intro}</p>
            {section.steps.map((step) => (
              <div key={step.label} className="flex flex-col gap-1">
                <p className="text-body text-ink">{step.prose}</p>
                <Equation label={step.label} tex={step.tex} />
              </div>
            ))}
            {section.outro ? <p className="text-body text-ink">{section.outro}</p> : null}
          </section>
        ))}
      </article>
    </>
  );
}
