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

    let modelWeight = 0.85;
    let metaWeight = 0.10;
    let webWeight = 0.05;

    if (cleanMeta >= 80) {
        metaWeight = 0.15;
        modelWeight = 0.75;
        webWeight = 0.10;
    }

    if (cleanWeb >= 80) {
        metaWeight = 0.15;
        modelWeight = 0.75;
        webWeight = 0.10;
    }

    const weightedScore = (cleanModel * modelWeight) + (cleanMeta * metaWeight) + (cleanWeb * webWeight);
    const finalScore = Math.round(clamp(weightedScore));

    let verdict = "Real Image"; // 0-20
    if (finalScore > 85) verdict = "AI Generater Image";
    else if (finalScore > 60) verdict = "High Likely AI";
    else if (finalScore > 40) verdict = "Suspicious / Likely AI";
    else if (finalScore > 20) verdict = "Real Image";

    const scoreBreakdown = [
        `Model signal: ${Math.round(cleanModel)}%`,
        `Metadata signal: ${Math.round(cleanMeta)}%`,
        `Web signal: ${Math.round(cleanWeb)}%`
    ].join(" | ");

    let reasoningDetail = "Combined evidence indicates this is likely a real image.";
    if (finalScore >= 85) {
        reasoningDetail = "ML model strongly indicates this image is AI generated.";
    } else if (finalScore >= 60) {
        reasoningDetail = "Combined score indicates a high likelihood of AI generation.";
    } else if (finalScore >= 40) {
        reasoningDetail = "Combined score is suspicious and leans toward AI generation.";
    } else if (finalScore >= 20) {
        reasoningDetail = "ML model strongly indicates this image is real.";
    }

    const reasoning = `Scores:\n${scoreBreakdown}\n\nReasoning:\n${reasoningDetail}`;

    return {
        finalScore,
        verdict,
        reasoning
    };
}
