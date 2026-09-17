import type { Experiment } from '../types';

interface EditorialCopy {
  question: string;
  result: string;
  explanation?: string;
  comparison?: string;
  why?: string;
}

/**
 * Public-facing prose for register rows whose source wording is useful for an
 * audit but too dense for the notebook interface. The source fields stay
 * unchanged and remain available on the experiment page.
 */
const editorialCopy:Record<string,EditorialCopy>={
  MT175:{
    question:'At a fixed group count, does the uniform partition always beat the architecture-aligned partition?',
    result:'Uniform partitions beat architecture-aligned partitions in all 20 count-matched cells. This is a synthesis of the underlying batch results.',
  },
  MT176:{
    question:'Does weightwise optimization still underperform after tuning its meta step size and removing clamp effects?',
    result:'The weightwise deficit persisted after meta-step tuning, so the clamp does not explain it. The fully tuned granularity ladder also identified its best setting.',
  },
  MT177:{
    question:'Across the tuned ladder, does meta-gradient sign agreement decrease as group count increases?',
    result:'The earlier accuracy pattern replicated, and the independent test confirmed the predicted decrease in sign agreement as group count increased.',
  },
  MT178:{
    question:'At matched group counts, does partition structure affect accuracy?',
    result:'Partition structure affected accuracy at nearly fixed group count. One registered check replicated, while the prediction that partition would not matter was refuted.',
  },
  MT179:{
    question:'What happens between the nodewise and weightwise ends of the granularity range?',
    result:'Accuracy rose monotonically across the fine-grained end of the ladder, by about 0.51 percentage points per decade, with no evidence of a knee.',
  },
  MT180:{
    question:'Does accuracy continue rising toward coarser groups, and can any chunk size beat the tuned ladder\'s best setting?',
    result:'Both registered checks were met. Accuracy continued rising toward the coarse end, while the within-batch comparison ended in a statistical tie.',
  },
  MT181:{
    question:'Does the partition effect persist in a directly measured count-matched comparison?',
    result:'The first direct count-matched comparison confirmed a +0.485 percentage-point partition gap, and the second registered check replicated.',
  },
  MT182:{
    question:'Which drives the partition gap: architecture alignment or the group-size distribution?',
    result:'The group-size distribution effect replicated. Alignment was consistent with a bounded null, which does not by itself refute an alignment effect.',
  },
  MT184:{
    question:'Does the scalar-to-layerwise gap survive meta-step tuning and removal of the singleton tail?',
    result:'The gap survived meta-step tuning, but it collapsed when the singleton tail was removed. None of the 12 arms was clamp-free, so the deconfounded comparison remains open.',
  },
  MT185:{
    question:'Can meta-gradient sign agreement predict accuracy at matched group counts?',
    result:'Meta-gradient sign agreement did not predict accuracy at matched group count. This proposed explanatory direction was therefore dropped.',
  },
  MT186:{
    question:'With new seeds, does the partition gap replicate, and does removing the singleton tail still eliminate it at an exact group count?',
    result:'The +0.727 percentage-point partition gap replicated with new seeds. After removing the singleton tail at an exact group count, the remaining gap was +0.011 percentage points.',
  },
  MT187:{
    question:'Does the partition gap scale with the fraction of singleton groups?',
    result:'The singleton-fraction law was withdrawn before data collection, and the cross-architecture test was untestable. No dose-response conclusion can be made.',
  },
  MT188:{
    question:'Do the measured gaps survive when the step-size clamp cannot bind?',
    result:'The main gap survived, but changing the clamp also changed the optimizer. The weightwise comparison remained underpowered, so the overall result is mixed.',
  },
  MT189:{
    question:'Can a fixed-count ladder identify a response to singleton-group fraction?',
    result:'The proposed fixed-count ladder was withdrawn before any run. It therefore did not test the intended singleton dose-response.',
  },
  MT190:{
    question:'Do the partition gap and the tail-removal mechanism transfer to ResNet-34?',
    result:'On ResNet-34, the partition gap was +0.666 percentage points and the tail-removed contrast remained near zero. The registered transfer and mechanism checks were met.',
  },
  MT191:{
    question:'Does the partition gap persist on CIFAR-100 under a registered stopping rule?',
    result:'On CIFAR-100, the partition gap was +1.640 percentage points. The result met the registered close-confirmation rule.',
  },
  MT192:{
    question:'Did the clamp-occupancy audit read the correct array?',
    result:'It did not. Three earlier scorers had read a summary array that hid full clipping or inverted its direction; the corrected audit identified the affected batches.',
  },
  MT193:{
    question:'Can one group-count slope correct comparisons across batches?',
    result:'The proposed cross-batch count-slope correction was deleted before it could be used. This row records the rejected analysis method.',
  },
  MT194:{
    question:'Is the gap caused by singleton groups or by BatchNorm?',
    result:'The GroupNorm comparison failed its commensurability requirement. Neither a transfer verdict nor a null verdict can be issued.',
  },
  MT195:{
    question:'Do singleton groups cause the reported non-monotonicity?',
    result:'Size-1 groups did not cause the reported non-monotonicity. The proposed general mechanism was refuted.',
  },
  MT196:{
    question:'Is the damage specific to singleton groups, graded by one-dimensional group size, or tensor-specific?',
    result:'One dose study was withdrawn and the other was blocked by a missing implementation patch. The graded-dose question remains open.',
  },
  MT197:{
    question:'Does the gap survive after tuning each arm independently in a nonbinding clamp range?',
    result:'The gap was unchanged at independently tuned optima in a provably nonbinding clamp range. This result is limited to ResNet-18 on CIFAR-10.',
  },
  MT198:{
    question:'Does the partition gap transfer from momentum SGD to an AdamW base optimizer?',
    result:'With six runs, the AdamW partition gap was +0.279 percentage points, with a 95% lower bound of +0.108. The registered scorer still classified the transfer as unresolved.',
  },
  MT199:{
    question:'On CIFAR-100, does the partition gap replicate, and does removing the singleton tail eliminate it?',
    result:'The numerical result was favorable, but no scorer had been registered before the runs. The experiment therefore has no registered outcome.',
  },
  MT200:{
    question:'Does the partition gap persist when the training budget is tripled?',
    result:'The partition gap remained positive at 300 epochs (+0.428 percentage points), so the mechanism survived three times the original training budget.',
  },
  MT201:{
    question:'Does the partition gap change under SGD and RMSProp base optimizers?',
    result:'The measured gaps were +1.035 percentage points with SGD and +0.973 with RMSProp. An analysis rule had been registered, but no outcome scorer, so the verdict remains open.',
  },
  MT202:{
    question:'Does the gap transfer to ResNet-50?',
    result:'The ResNet-50 cell did not produce a registered verdict. Transfer to the bottleneck-block architecture remains open.',
  },
  MT203:{
    question:'Does the partition gap persist with Adam or RMSProp as the meta optimizer instead of Lion?',
    result:'The meta-optimizer comparison was void: every logged run still used Lion. The batch can only be treated as another Lion replicate.',
  },
  MT204:{
    question:'Which feature of the group-size distribution explains the partition gap?',
    result:'The design matrix had rank 3 and could identify at most two statistics. The responsible group-size feature is not identifiable from this corpus.',
  },
  MT205:{
    question:'Can measured configuration features predict the gap?',
    result:'None of the measured features predicted the gap out of sample better than its mean. The predictive framing was rejected.',
  },
  MT206:{
    question:'Does combining an AdamW base optimizer with an RMSProp meta optimizer reduce the partition gap?',
    result:'The planned second-moment comparison was void because every logged run repeated the same meta-optimizer setting. The proposed mechanism remained untested.',
  },
  MT207:{
    question:'Does the base-optimizer effect replicate in an independent SGD and RMSProp batch?',
    result:'The effect replicated at both base-optimizer levels. The heterogeneity statistic stayed within its registered bound (Q ≤ 3.841).',
  },
  MT208:{
    question:'Does second-moment normalization reduce the gap?',
    result:'At the AdamW base and RMSProp meta corner, the gap remained at least +0.55. Second-moment normalization did not reduce it, so the proposed mechanism was refuted.',
  },
  MT209:{
    question:'Can differences in the partition gap be attributed to the base optimizer?',
    result:'The base optimizer remains only a candidate moderator because it was not separated from the submission batch. A causal attribution is not supported.',
  },
  MT210:{
    question:'Does the alignment null replicate at twice the power when permutation draw and run seed are separated?',
    result:'The alignment null replicated. The higher-power check remained underpowered, and an unexpected run-seed effect is unresolved.',
  },
  MT211:{
    question:'After repairing the confounded run, is the partition gap flat from 100 to 300 epochs?',
    result:'The partition gap declined by 0.238 percentage points from 100 to 300 epochs, although it remained positive at 300 epochs. The registered flatness hypothesis failed.',
  },
  MT212:{
    question:'Does the scalar-to-layerwise gap persist without BatchNorm?',
    result:'The scalar-to-layerwise gap persisted on GroupNorm ResNet-18: +37.38 percentage points at 100 epochs. The gap therefore does not require BatchNorm.',
    explanation:'The measured gap cleared the registered threshold, and the training result agreed. This remains a 100-epoch result from one GroupNorm ResNet-18 setting.',
    comparison:'Scalar versus layerwise MetaOptimize on GroupNorm ResNet-18 at 100 epochs.',
    why:'To test whether the scalar-to-layerwise gap depends on BatchNorm or persists when BatchNorm is replaced by GroupNorm.',
  },
  MT213:{
    question:'After removing residual connections, which BatchNorm scale tensors still carry the gap?',
    result:'The two-tensor set improved accuracy by +55.85 percentage points over its matched control, while the single-tensor arm stayed at the floor. The registered mechanism accounts were only partly supported.',
    explanation:'The paired isolation supports a carrier effect, but the single-tensor result and missed registered branches prevent a complete success verdict.',
    comparison:'Two nominated carrier tensors versus one carrier and matched controls on PlainNet.',
    why:'To identify which BatchNorm scale tensors carry the gap after residual connections are removed.',
  },
  MT214:{
    question:'Does the VGG isolation rescue persist to 328 epochs?',
    result:'The VGG isolation gain was unchanged at 328 epochs (ρ = 1.0001), after the complementary group had pinned. All six registered predictions were within their bands.',
    explanation:'The long run preserved the rescue after the complement pinned. Carrier identity and step-size magnitude remain confounded.',
    comparison:'The isolated VGG carrier group and its controls at 100 versus 328 epochs.',
    why:'To determine whether the isolation rescue is transient or persists long after the complementary group reaches its step-size floor.',
  },
  MT215:{
    question:'Without augmentation on CIFAR-100, does tuned SGD still outperform the meta-optimized method?',
    result:'Without augmentation, tuned SGD reached 65.06% versus 52.88% for the best method cell, a +12.18 percentage-point gap. The SGD optimum was inside the tested learning-rate grid.',
    explanation:'The registered deficit remained after tuning both methods. The result covers one baseline family, one granularity setting, ResNet-18, and CIFAR-100.',
    comparison:'A tuned SGD learning-rate ladder versus a tuned MetaOptimize step-size ladder on ResNet-18 and CIFAR-100 without augmentation.',
    why:'To test whether tuning or data augmentation had hidden the baseline advantage on CIFAR-100.',
  },
  MT216:{
    question:'Does the carrier-isolation mechanism transfer from BatchNorm to GroupNorm?',
    result:'On GroupNorm ResNet-18, the three nominated carriers rescued performance while the matched control did not; tensor 50 alone also rescued.',
    explanation:'Carrier identity transferred to GroupNorm at 100 epochs. Step-size magnitude, possible mistuning, and longer-run persistence were not separated.',
    comparison:'Three nominated GroupNorm carriers and tensor 50 alone versus matched controls.',
    why:'To test whether the carrier-isolation mechanism transfers from BatchNorm to GroupNorm.',
  },
  MT217:{
    question:'On PlainNet, can layer4.1.bn2.weight alone produce the isolation rescue?',
    result:'On PlainNet, isolating layer4.1.bn2.weight improved accuracy by +52.14 percentage points over its twin control. The tensor alone reproduced the rescue.',
    explanation:'The single tensor carried the rescue in this decomposition. Tensor identity and step-size magnitude remain confounded, and the result is limited to 100 epochs.',
    comparison:'layer4.1.bn2.weight alone versus a matched twin-tensor control on PlainNet.',
    why:'To test whether one tensor is sufficient for the rescue after residual connections are removed.',
  },
  MT218:{
    question:'On PlainNet, does the rescue require a separate step size for layer4.1.bn2.weight, or only removal of its meta-gradient vote?',
    result:'Removing tensor 50\'s vote while keeping the shared step size did not rescue the model. Giving it a separate step size did; injecting a large twin-tensor vote caused a partial collapse of +34.65 percentage points.',
    explanation:'A separate step size was necessary in this cell, and an injected carrier-sized vote was sufficient to cause damage. The injected arm was only a partial collapse and missed its registered band by 0.14 percentage points.',
    comparison:'Separate-step-size, vote-removal, and injected-vote interventions for tensor 50 on PlainNet.',
    why:'To distinguish the effect of tensor 50\'s own step size from the effect of its meta-gradient vote on the complementary group.',
  },
  MT219:{
    question:'Does the GroupNorm isolation rescue persist to 430 epochs?',
    result:'The GroupNorm isolation rescue persisted to 430 epochs (ρ = 1.2235) after the complementary group pinned. The single-tensor rescue also persisted.',
    explanation:'Every arm met the registered long-run account. The isolated arm finished 3.02 percentage points below the layerwise reference, and the level settled before the complement pinned.',
    comparison:'GroupNorm carrier-isolation arms at 100 and 430 epochs, including the single-tensor arm.',
    why:'To determine whether the GroupNorm rescue persists after the complementary group reaches its step-size floor.',
  },
  MT220:{
    question:'On PlainNet, can muting the full negative-vote coalition rescue the scalar model while all tensors share one step size?',
    result:'Muting tensor 50 and the 20-tensor negative-vote coalition left accuracy at the scalar baseline: 11.33% versus 11.57%. A shared step size did not produce a rescue.',
    explanation:'The result supports the need for tensor 50 to have its own step size. One control missed its registered band by 0.18 percentage points, and the negative vote was re-carried by the next tier.',
    comparison:'A shared step size with tensor 50 and the negative-vote coalition muted, compared with scalar and matched controls.',
    why:'To test whether removing the full negative vote can rescue PlainNet without giving tensor 50 a separate step size.',
  },
  MT221:{
    question:'Does a carrier tensor\'s injected vote cause a threshold effect or a graded loss?',
    result:'The retained share of the rescue fell gradually from 77.2% at K = 13 to 25.6% at K = 2000. The effect was graded rather than a fixed threshold.',
    explanation:'No registered account predicted the full pattern, two readings were close to their decision bars, and the K = 13 arm was still improving at 100 epochs.',
    comparison:'Five injected vote weights, from K = 13 to K = 2000, compared with the isolated-tensor rescue and scalar baseline.',
    why:'To determine whether an injected carrier vote causes a sharp threshold or a graded loss of the rescue.',
  },
  MT222:{
    question:'On PlainNet, is the isolated tensor\'s step-size magnitude sufficient to switch between rescue and collapse?',
    result:'With tensor 50\'s vote removed, a large replayed step size collapsed accuracy to 11.13%, while holding it at the floor preserved the 65.46% rescue.',
    explanation:'All nine registered bands were met, showing that the tensor\'s own step-size magnitude is sufficient at this cell. Direct damage and damage through the complementary group remain confounded.',
    comparison:'Tensor 50 held at small, shared, large, or replayed step sizes while its vote is excluded from the complementary group.',
    why:'To test whether tensor 50\'s own step-size magnitude can switch PlainNet between rescue and collapse.',
  },
  MT223:{
    question:'Do the PlainNet rescue and the graded vote-weight effect persist to 300 epochs?',
    result:'At 300 epochs, the K = 13 arm remained 15.95 percentage points above K = 33, and the isolated-tensor rescue remained stable (ρ = 1.0012) after pinning.',
    explanation:'The graded pattern is not a 100-epoch delay, and all eight registered bands were met. Some level and pin readings remain close to their decision boundaries.',
    comparison:'The isolated-tensor rescue and K = 13 and K = 33 vote-weight arms at 100 versus 300 epochs.',
    why:'To determine whether the graded vote-weight pattern is a temporary delay or a stable long-run level.',
  },
  MT224:{
    question:'On PlainNet, does the isolated tensor\'s large step size cause failure directly or by collapsing the complementary group?',
    result:'Both routes mattered. A large step size on tensor 50 with the complementary group held open collapsed accuracy to 11.49%, while a collapsed complementary group with tensor 50 held small reached 41.56%.',
    explanation:'The direct route produced a full collapse and the complementary-group route produced a partial loss. No registered account predicted the full graded pattern; the forced paths were open-loop and tested at one setting.',
    comparison:'A 2 × 2 intervention: tensor 50 at a small or large step size, with the complementary group forced onto an open or collapsed path.',
    why:'To separate damage caused directly by tensor 50\'s large step size from damage mediated by early collapse of the complementary group.',
  },
  MT225:{
    question:'Does the small-step-size carrier rescue transfer from PlainNet to ResNet-18?',
    result:'On ResNet-18, a large carrier-group step size reduced accuracy to 50.23%, between the scalar baseline (22.95%) and isolated arm (70.04%).',
    explanation:'The registered graded-transfer account met every band. Network, dose, and held-set differences still limit comparison with the PlainNet experiment.',
    comparison:'The ResNet-18 carrier group on small, large, and replayed step-size schedules, compared with scalar and isolated controls.',
    why:'To test whether the small-step-size carrier rescue found on PlainNet transfers to ResNet-18.',
  },
  MT226:{
    question:'On ResNet-18, how do carrier step-size dose and complementary-group collapse contribute to accuracy loss?',
    result:'On ResNet-18, the larger PlainNet dose drove accuracy to 19.38% versus a 23.06% scalar baseline, while the smaller dose produced 49.63%. Holding the complementary path recovered 8.56 percentage points only at the smaller dose.',
    explanation:'Dose had a large effect and the complementary route contributed at the smaller dose. No registered account fit every band, and floor saturation limits the route comparison at the larger dose.',
    comparison:'Two carrier-group doses crossed with a free or forced complementary path on ResNet-18.',
    why:'To separate the effect of carrier step-size dose from the effect of earlier collapse in the complementary group.',
  },
  MT227:{
    question:'On PlainNet, how do step-size dose and timing for the isolated tensor affect the collapse?',
    result:'On PlainNet, accuracy fell across the three tested dose schedules: 55.86%, 21.84%, then 11.21%. Applying only the early or late half of the largest schedule produced 18.37% and 27.71%.',
    explanation:'The response was graded by both dose and timing, but no registered account fit every band. The study used one open-loop window pair and does not establish a threshold, functional form, or necessary window.',
    comparison:'Three dose schedules plus early-half and late-half windows for tensor 50, with the complementary group fixed to the rescued path.',
    why:'To measure how collapse changes with step-size dose and to locate which part of the largest schedule causes the damage.',
  },
};

const markerByOutcome={success:'Goal met:',fail:'Goal missed:',mixed:'Mixed:',unresolved:'Open:'} as const;

function registeredRationale(experiment:Experiment){
  const marker=experiment.kind==='method-check'?'Method check:':experiment.outcome?markerByOutcome[experiment.outcome]:'';
  const markerIndex=marker?experiment.reason.indexOf(marker):-1;
  const prose=(markerIndex>=0?experiment.reason.slice(markerIndex+marker.length):experiment.reason).trim();
  return prose.replace(/^[A-Z0-9][A-Z0-9 _+./'"=><()-]{1,100}:\s*/,'').trim()||experiment.reason;
}

export function experimentQuestion(experiment:Experiment){return editorialCopy[experiment.id]?.question||experiment.goal;}
export function experimentResult(experiment:Experiment){return editorialCopy[experiment.id]?.result||registeredRationale(experiment)||experiment.result;}
export function experimentReason(experiment:Experiment){return editorialCopy[experiment.id]?.explanation||registeredRationale(experiment)||experiment.reason;}
export function experimentComparison(experiment:Experiment){return editorialCopy[experiment.id]?.comparison||experiment.comparison;}
export function experimentWhy(experiment:Experiment){return editorialCopy[experiment.id]?.why||experiment.why;}
export function hasEditorialCopy(experiment:Experiment){return Boolean(editorialCopy[experiment.id]);}
