import express,{type Request, type Response} from "express"
import multer from "multer";
import { checkWebScore } from "../service/visualSearch.service.js";
import { checkMetaData } from "../utils/metadata.extraction.js";
import { fuseScores } from "../service/score.service.js";
import { predictWithPythonModel } from "../service/pythonModel.service.js";

const router = express.Router();
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 5 * 1024 * 1024, // Limit to 5MB
  },
  fileFilter: (req, file, cb) => {
    const allowedTypes = ["image/jpeg", "image/png", "image/webp", "image/jpg"];
    if (allowedTypes.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(new Error("Only JPG, PNG, and WEBP are supported"));
    }
  }
});


router.get("/health", async(req: Request, res:Response) => {
    res.status(200).json("All OK")
})

router.post("/scan", upload.single("image"), async(req: Request, res:Response) => {
    try {
        if(!req.file){
            return res.status(400).json("No Image Uploaded")
        }

        const imageBuffer = req.file.buffer;


        // ------------- 1. Metadata Result &  --------------
        const metaScore = await checkMetaData(imageBuffer);


        // ------------- 2. Visual Search API --------------
        const webScore = await checkWebScore(imageBuffer);


        // ------------- 3. Model inference via embedded Python process ---------------
        const modelResponse = await predictWithPythonModel(imageBuffer, req.file.mimetype);


        // ------------- 4. Scoring ---------------
        const modelScore = modelResponse.ai_score;

        const fusionResult = fuseScores({
            modelScore,
            metaScore: metaScore.analysis.score,
            webScore,
            override: metaScore.analysis.override
        })
        console.log("MODEL RAW RESPONSE:", modelResponse);
        
        res.json({
          final_score: fusionResult.finalScore,
          verdict: fusionResult.verdict,
          reasoning: fusionResult.reasoning,
          breakdown: {
            model: Math.round(modelScore),
            metadata: Math.round(metaScore.analysis.score),
            web: Math.round(webScore)
          }
        })


    } catch (error) {
        console.error("Scan failed:", error);
        res.status(500).json("System failure");
    }
})


export default router;
