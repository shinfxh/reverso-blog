const paperUrl = "https://arxiv.org/abs/2602.17634";
const codeUrl = "https://github.com/shinfxh/reverso";
const modelsUrl = "https://huggingface.co/shinfxh/reverso";

export default function Home() {
  return (
    <main>
      <header>
        <p className="label">Paper summary</p>
        <h1 className="paper-title">
          <strong>Reverso</strong>
          <span>
            Efficient Time Series Foundation Models for zero-shot forecasting
          </span>
        </h1>
        <p className="authors">
          Xinghong Fu<sup>1</sup>, Yanhong Li<sup>2</sup>, Georgios
          Papaioannou<sup>3</sup>, and Yoon Kim<sup>1</sup>
          <br />
          <span>
            <sup>1</sup>Massachusetts Institute of Technology;{" "}
            <sup>2</sup>Allen Institute for AI; <sup>3</sup>Qube Research &amp;
            Technologies
          </span>
        </p>
        <p>
          <time dateTime="2026-07">July 2026</time>
        </p>
        <p className="links">
          <a href="demo/">Live demo</a> · <a href={paperUrl}>Paper</a> ·{" "}
          <a href={codeUrl}>Code</a> ·{" "}
          <a href={modelsUrl}>Models</a> ·{" "}
          <a href="mailto:fxh@mit.edu">Correspondence</a>
        </p>
      </header>

      <section aria-labelledby="summary">
        <h2 id="summary">Summary</h2>
        <p>
          Time series foundation models are often scaled to hundreds of
          millions—or even billions—of parameters. Reverso asks a simpler
          question: how small can a general-purpose forecasting model be while
          remaining competitive?
        </p>
        <p>
          Reverso combines multi-scale inputs, long convolutions, DeltaNet
          layers, and an attention-based decoder. The largest model has only{" "}
          <strong>2.6 million parameters</strong>, yet reaches a{" "}
          <strong>0.706 MASE</strong> on the full Gift-Eval benchmark and
          performs strongly across forecasting, classification, anomaly
          detection, imputation, and probabilistic forecasting.
        </p>
      </section>

      <nav aria-label="Contents">
        <h2>Contents</h2>
        <ol>
          <li><a href="#motivation">Motivation</a></li>
          <li><a href="#architecture">Architecture</a></li>
          <li><a href="#training">Training recipe</a></li>
          <li><a href="#forecasting">Forecasting results</a></li>
          <li><a href="#downstream">Beyond forecasting</a></li>
          <li><a href="#efficiency">Efficiency</a></li>
          <li><a href="#limitations">Limitations</a></li>
        </ol>
      </nav>

      <section id="motivation">
        <h2>1. Motivation</h2>
        <p>
          Foundation models make it possible to forecast an unfamiliar time
          series without task-specific training. But most progress has followed
          the scaling playbook from language and vision: train a larger model on
          more data. That makes strong forecasters harder to deploy on-device,
          at the edge, or anywhere memory and latency matter.
        </p>
        <p>
          Reverso explores the opposite direction. It keeps the architecture
          compact, uses efficient sequence-mixing primitives, and adds practical
          training and inference strategies that improve accuracy without
          requiring a large transformer.
        </p>
        <table>
          <caption>Reverso model family</caption>
          <thead>
            <tr>
              <th>Model</th>
              <th>Parameters</th>
              <th>Layers</th>
              <th>Width</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>Reverso-Nano</td>
              <td>200K</td>
              <td>2</td>
              <td>32</td>
            </tr>
            <tr>
              <td>Reverso-Small</td>
              <td>550K</td>
              <td>4</td>
              <td>64</td>
            </tr>
            <tr>
              <td>Reverso</td>
              <td>2.6M</td>
              <td>8</td>
              <td>128</td>
            </tr>
          </tbody>
        </table>
      </section>

      <section id="architecture">
        <h2>2. Architecture</h2>
        <p>
          A Reverso block alternates two efficient ways of mixing information
          across time: long convolutions capture broad patterns in parallel,
          while DeltaNet layers maintain a recurrent state for sequential
          dependencies. Each sequence mixer is followed by a small MLP for
          channel mixing.
        </p>
        <p>
          The input is represented at four temporal scales. Each scale looks
          farther back in time through progressively stronger downsampling, so
          the model receives up to 16,384 historical points while processing a
          sequence of length 2,048. An attention-based decoder then predicts 48
          future points at a time.
        </p>
        <figure className="architecture-figure">
          <img
            src="assets/architecture.png"
            alt="Reverso architecture with multi-scale input channels, alternating long convolution and DeltaNet blocks, and an attention decoder."
            loading="lazy"
          />
          <figcaption>
            The Reverso architecture: multi-scale input, hybrid sequence
            mixing, and an attention decoder.
          </figcaption>
        </figure>
        <ol className="method-list">
          <li>
            <strong>Multi-scale context.</strong> Four downsampled views expose
            both recent detail and long-range structure.
          </li>
          <li>
            <strong>Hybrid sequence mixing.</strong> Long convolutions and
            DeltaNet layers alternate throughout the model.
          </li>
          <li>
            <strong>Attention decoder.</strong> Learned output queries attend to
            the encoded history and predict the next 48 points.
          </li>
        </ol>
      </section>

      <section id="training">
        <h2>3. Training recipe</h2>
        <p>
          The models are pretrained on heterogeneous real and synthetic time
          series. The data pipeline combines standard augmentations—including
          mixup, downsampling, censoring, and amplitude modulation—with
          Gaussian-process, spike-process, and TSI synthetic sequences.
        </p>
        <figure>
          <img
            src="assets/training-pipeline.png"
            alt="Reverso data pipeline showing real datasets, augmentation methods, and synthetic generators."
            loading="lazy"
          />
          <figcaption>
            Real and synthetic time series are combined in a single pretraining
            pipeline.
          </figcaption>
        </figure>
        <p>
          Full training takes roughly 10, 20, and 40 H100-hours for
          Reverso-Nano, Reverso-Small, and Reverso respectively. An FFT-based
          downsampling rule also compresses very long seasonal patterns when
          their period exceeds the model context.
        </p>
      </section>

      <section id="forecasting">
        <h2>4. Forecasting results</h2>
        <p>
          Gift-Eval spans 97 forecasting tasks from 23 datasets across seven
          domains. Reverso reaches 0.706 overall MASE with 2.6M parameters,
          outperforming similarly sized models and closely tracking much larger
          systems.
        </p>
        <table>
          <caption>Full Gift-Eval comparison</caption>
          <thead>
            <tr>
              <th>Model</th>
              <th>Parameters</th>
              <th>MASE ↓</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>FlowState-r1.1</td>
              <td>18M</td>
              <td>0.701</td>
            </tr>
            <tr>
              <td>TimesFM-2.5</td>
              <td>200M</td>
              <td>0.705</td>
            </tr>
            <tr>
              <td>Reverso</td>
              <td>2.6M</td>
              <td>0.706</td>
            </tr>
            <tr>
              <td>Xihe-Max</td>
              <td>1.5B</td>
              <td>0.711</td>
            </tr>
          </tbody>
        </table>
        <p>
          On the 21 long-sequence Gift-Eval datasets with all three horizon
          lengths, Reverso has the best average MASE in the reported comparison:
          0.691. It is especially strong at medium and long horizons despite
          forecasting autoregressively in 48-point chunks.
        </p>
        <table>
          <caption>Long-sequence Gift-Eval tasks</caption>
          <thead>
            <tr>
              <th>Model</th>
              <th>Short</th>
              <th>Medium</th>
              <th>Long</th>
              <th>Average</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>PatchTST-FM</td>
              <td><strong>0.616</strong></td>
              <td>0.722</td>
              <td>0.745</td>
              <td>0.694</td>
            </tr>
            <tr>
              <td>FlowState-r1.1</td>
              <td>0.633</td>
              <td>0.720</td>
              <td><strong>0.736</strong></td>
              <td>0.696</td>
            </tr>
            <tr>
              <td>Reverso</td>
              <td>0.634</td>
              <td><strong>0.699</strong></td>
              <td>0.741</td>
              <td><strong>0.691</strong></td>
            </tr>
          </tbody>
        </table>
        <p>
          Reverso also transfers well to the LTSF benchmark, averaging 0.322 MAE
          across six datasets and four prediction horizons—competitive with
          models tens or hundreds of times larger.
        </p>
      </section>

      <section id="downstream">
        <h2>5. Beyond forecasting</h2>
        <p>
          Although trained primarily as a univariate forecaster, Reverso learns
          representations that transfer to other time-series tasks.
        </p>
        <table>
          <caption>Classification accuracy with a frozen Reverso encoder</caption>
          <thead>
            <tr>
              <th>Model</th>
              <th>Parameters</th>
              <th>UCR</th>
              <th>Multivariate UEA</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Reverso</strong></td>
              <td>2.6M</td>
              <td><strong>81%</strong></td>
              <td><strong>74%</strong></td>
            </tr>
            <tr>
              <td>TiRex</td>
              <td>30M</td>
              <td>81%</td>
              <td>74%</td>
            </tr>
            <tr>
              <td>Chronos-Bolt</td>
              <td>205M</td>
              <td>79%</td>
              <td>74%</td>
            </tr>
          </tbody>
        </table>
        <p>
          On zero-shot UCR anomaly detection, Reverso detects 88, 126, and 188
          labeled anomalies when the top 1%, 3%, and 10% of forecast-error
          segments are flagged. It also achieves the best reported imputation
          MSE on all seven tested datasets, and its pretrained backbone supports
          competitive probabilistic forecasts after training a
          quantile-regression decoder.
        </p>
        <p className="note">
          These downstream evaluations use lightweight task-specific protocols:
          a frozen encoder plus random forest for classification, forecast error
          for anomaly detection, and fine-tuned heads for imputation and
          probabilistic forecasting.
        </p>
      </section>

      <section id="efficiency">
        <h2>6. Efficiency</h2>
        <h3>Parameter efficiency</h3>
        <p>
          Reverso reaches competitive zero-shot forecasting accuracy with only
          2.6 million parameters. It is roughly 77 times smaller than
          TimesFM-2.5 and more than 570 times smaller than Xihe-Max, while
          remaining on the Gift-Eval and LTSF performance-efficiency frontiers.
        </p>
        <figure className="efficiency-figure">
          <img
            src="assets/gift-eval-pareto.png"
            alt="Gift-Eval MASE plotted against model parameter count. Reverso sits on the performance-efficiency Pareto frontier at 2.6 million parameters."
            loading="lazy"
          />
          <figcaption>
            Gift-Eval performance versus parameter count. Lower MASE is better.
          </figcaption>
        </figure>
        <figure>
          <img
            src="assets/ltsf-pareto.png"
            alt="LTSF average MAE plotted against model parameter count, with Reverso among the strongest small models."
            loading="lazy"
          />
          <figcaption>
            Zero-shot LTSF performance versus parameter count. Lower MAE is
            better.
          </figcaption>
        </figure>

        <h3>Latency efficiency</h3>
        <p>
          The compact parameter count translates into practical inference
          efficiency. On a single H100, Reverso variants achieve lower latency
          and use less peak memory than much larger time series foundation
          models while staying on the performance frontier.
        </p>
        <figure className="efficiency-figure">
          <div className="figure-grid">
            <img
              src="assets/latency.png"
              alt="Inference latency comparison on Gift-Eval."
              loading="lazy"
            />
            <img
              src="assets/memory.png"
              alt="Peak inference memory comparison on Gift-Eval."
              loading="lazy"
            />
          </div>
          <figcaption>
            Gift-Eval inference latency and peak memory for single-sample
            inference on one H100.
          </figcaption>
        </figure>
      </section>

      <section id="limitations">
        <h2>7. Limitations</h2>
        <p>Several limitations remain:</p>
        <ul>
          <li>
            Reverso is trained primarily as a univariate forecaster and does not
            explicitly model cross-channel dependence.
          </li>
          <li>
            Short-sequence performance still trails the strongest large models
            on some tasks.
          </li>
          <li>
            The base model produces point forecasts; probabilistic forecasts
            require an adapted decoder objective.
          </li>
          <li>
            Some efficiency measurements use an H100 and may differ on
            edge-class hardware.
          </li>
        </ul>
      </section>

      <section id="citation">
        <h2>Citation</h2>
        <pre>
          <code>{`@misc{fu2026reverso,
  title   = {Reverso: Efficient Time Series Foundation Models
             for Zero-shot Forecasting},
  author  = {Fu, Xinghong and Li, Yanhong and Papaioannou,
             Georgios and Kim, Yoon},
  year    = {2026},
  eprint  = {2602.17634},
  archivePrefix = {arXiv}
}`}</code>
        </pre>
      </section>

      <footer>
        <p>
          <a href={paperUrl}>Read the complete paper</a> ·{" "}
          <a href={codeUrl}>Explore the code</a> ·{" "}
          <a href={modelsUrl}>Download the model</a>
        </p>
        <p>
          Correspondence: <a href="mailto:fxh@mit.edu">fxh@mit.edu</a>
        </p>
      </footer>
    </main>
  );
}
