import { Equation } from "../components/Equation";
import { Card, PageHeader } from "../components/ui";

const SECTIONS: { title: string; text: string; equations: { label: string; tex: string }[] }[] = [
  {
    title: "The robustness checker",
    text: "A headline compares a metric between two groups. The checker recomputes the difference inside strata of each stratifier and combines them with Mantel-Haenszel weights; the verdict reverses when the adjusted difference changes sign.",
    equations: [
      {
        label: "Mantel-Haenszel adjusted difference",
        tex: String.raw`\Delta_{\text{adj}} = \frac{\sum_k w_k \Delta_k}{\sum_k w_k}, \qquad w_k = \frac{n_{T,k}\, n_{R,k}}{n_{T,k} + n_{R,k}}`,
      },
    ],
  },
  {
    title: "Calendar effects",
    text: "Daily counts follow a negative binomial with a log link on day of week, day-of-month bins, sum-coded months and calendar periods; intervals come from a client bootstrap.",
    equations: [
      {
        label: "Negative-binomial calendar model",
        tex: String.raw`\begin{aligned} \log \mu_t &= \beta_0 + \sum_d \gamma_d [\text{dow}_t = d] + \sum_b \delta_b [\text{bin}_t = b] \\ &\quad + \sum_m \theta_m c_{m,t} + \sum_p \lambda_p [t \in p], \\ \operatorname{Var}(Y_t) &= \mu_t + \alpha \mu_t^2 \end{aligned}`,
      },
    ],
  },
  {
    title: "The Markov chain and the simulator",
    text: "Each row of the transition matrix has a Dirichlet posterior; every derived quantity is computed on posterior draws, and a policy is projected with the same draws as its baseline.",
    equations: [
      { label: "Dirichlet posterior", tex: String.raw`P_{i\cdot} \mid N \sim \operatorname{Dirichlet}(1 + N_{i1}, \dots, 1 + N_{iK})` },
      { label: "Mean first-passage times", tex: String.raw`Z = (I - P + \mathbf{1}\pi)^{-1}, \qquad m_{ij} = \frac{z_{jj} - z_{ij}}{\pi_j}` },
      { label: "Projection", tex: String.raw`x_{h+1} = x_h P^{(s)}, \qquad L_h = x_h \cdot \rho` },
    ],
  },
  {
    title: "Amounts: empirical Bayes",
    text: "A client's typical log amount is shrunk toward its category's; a robust z of each amount flags the unusual ones.",
    equations: [
      { label: "Shrunk mean", tex: String.raw`\hat m_i = \frac{t^2 m_i + (s^2/n_i)\,\mu}{t^2 + s^2/n_i}` },
      { label: "Robust z", tex: String.raw`z = \frac{\log a - \hat m_i}{1.4826 \cdot \operatorname{MAD}}` },
    ],
  },
  {
    title: "Leading digits",
    text: "Benford's law, read with the mean absolute deviation and Nigrini's conformity bands.",
    equations: [
      { label: "Benford's law", tex: String.raw`P(D_1 = d) = \log_{10}\!\left(1 + \tfrac{1}{d}\right)` },
      { label: "Mean absolute deviation", tex: String.raw`\text{MAD} = \frac{1}{K}\sum_k \lvert \hat p_k - p_k \rvert` },
    ],
  },
];

export function MethodsPage() {
  return (
    <>
      <PageHeader
        title="Methods"
        lead="The equations behind the pages. The full derivations, assumptions and limitations are in docs/methods in the repository."
      />
      <div className="flex flex-col gap-4">
        {SECTIONS.map((section) => (
          <Card key={section.title} title={section.title} subtitle={section.text}>
            {section.equations.map((equation) => (
              <Equation key={equation.label} label={equation.label} tex={equation.tex} />
            ))}
          </Card>
        ))}
      </div>
    </>
  );
}
