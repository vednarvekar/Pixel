interface FusionInput {
    modelScore: number;
    metaScore: number; 
    webScore: number;
    override: boolean;
}

function clamp(value: number, min = 0, max = 100) {
    return Math.max(min, Math.min(max, value));
}

export function fuseScores(input: FusionInput) {
    const { modelScore, metaScore, webScore, override } = input;

    if (override == true) {
        return {
            finalScore: 100, // Change from 1.0 to 100
            verdict: "Definite AI (C2PA Verified)",
            reasoning: "C2PA provenance metadata cryptographically confirms AI generation."
        }
    }

    const cleanModel = clamp(modelScore);
    const cleanMeta = clamp(metaScore);
    const cleanWeb = clamp(webScore);

    let modelWeight = 0.65;
    let metaWeight = 0.20;
    let webWeight = 0.15;

    if (cleanMeta >= 80) {
        metaWeight = 0.30;
        modelWeight = 0.55;
        webWeight = 0.15;
    }

    if (cleanWeb >= 80) {
        webWeight = 0.25;
        modelWeight = 0.55;
        metaWeight = 0.20;
    }

    const weightedScore = (cleanModel * modelWeight) + (cleanMeta * metaWeight) + (cleanWeb * webWeight);
    const maxSignalGap = Math.max(cleanModel, cleanMeta, cleanWeb) - Math.min(cleanModel, cleanMeta, cleanWeb);

    // High disagreement usually means uncertain evidence, so avoid overconfident extremes.
    const consistencyPenalty = maxSignalGap > 45 ? Math.round((maxSignalGap - 45) * 0.25) : 0;
    const rawScore = clamp(weightedScore - consistencyPenalty);
    
    const finalScore = Math.round(rawScore);

    let verdict = "Real Image"; // 0-20
    if (maxSignalGap > 55 && finalScore >= 35 && finalScore <= 70) {
        verdict = "Inconclusive / Needs Manual Review";
    }
    if (finalScore > 85) verdict = "High Probability AI";
    else if (finalScore > 70) verdict = "Likely AI";
    else if (finalScore > 40) verdict = "Suspicious / Likely AI";
    else if (finalScore > 20) verdict = "Likely Real";

    const reasoning = [
      `Model signal: ${Math.round(cleanModel)}%.`,
      `Metadata signal: ${Math.round(cleanMeta)}%.`,
      `Web signal: ${Math.round(cleanWeb)}%.`,
      maxSignalGap > 45
        ? "Signals disagree strongly, confidence is reduced."
        : "Signals are reasonably consistent."
    ].join(" ");

    return {
        finalScore,
        verdict,
        reasoning
    };
}
