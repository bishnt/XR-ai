import React from 'react';

interface PredictionResult {
    class: string;
    confidence: number;
    probabilities: {
        NORMAL: number;
        PNEUMONIA: number;
        COVID: number;
        TB: number;
    };
}

interface ResultsCardProps {
    prediction: PredictionResult;
    processingTime?: number;
}

const ResultsCard: React.FC<ResultsCardProps> = ({ prediction, processingTime }) => {
    // Color scheme based on condition severity
    const getStatusColor = (className: string) => {
        switch (className) {
            case 'NORMAL':
                return 'text-white';
            case 'PNEUMONIA':
                return 'text-gray-300';
            case 'COVID':
                return 'text-gray-300';
            case 'TB':
                return 'text-gray-300';
            default:
                return 'text-white';
        }
    };

    // Format confidence as percentage
    const formatConfidence = (value: number) => {
        return `${(value * 100).toFixed(1)}%`;
    };

    // Get all probabilities in descending order
    const sortedProbabilities = Object.entries(prediction.probabilities)
        .sort((a, b) => b[1] - a[1])
        .map(([className, prob]) => ({
            class: className,
            probability: prob,
            isHighest: className === prediction.class
        }));

    return (
        <div className="w-full max-w-2xl mx-auto bg-black border border-white/20 p-8 rounded-sm">
            {/* Main Prediction */}
            <div className="text-center mb-8">
                <h2 className="text-sm uppercase tracking-widest text-white/60 mb-2">
                    Analysis Result
                </h2>
                <h1 className={`text-5xl font-light mb-2 ${getStatusColor(prediction.class)}`}>
                    {prediction.class}
                </h1>
                <p className="text-2xl text-white/80">
                    {formatConfidence(prediction.confidence)} confident
                </p>
            </div>

            {/* Divider */}
            <div className="w-full h-px bg-white/10 mb-8" />

            {/* Detailed Probabilities */}
            <div className="space-y-4">
                <h3 className="text-xs uppercase tracking-widest text-white/60 mb-4">
                    Detailed Probabilities
                </h3>

                {sortedProbabilities.map(({ class: className, probability, isHighest }) => (
                    <div key={className} className="space-y-2">
                        {/* Class Name and Percentage */}
                        <div className="flex justify-between items-baseline">
                            <span className={`text-sm ${isHighest ? 'text-white font-normal' : 'text-white/60'
                                }`}>
                                {className}
                            </span>
                            <span className={`text-sm tabular-nums ${isHighest ? 'text-white' : 'text-white/60'
                                }`}>
                                {formatConfidence(probability)}
                            </span>
                        </div>

                        {/* Progress Bar */}
                        <div className="w-full h-1 bg-white/10 rounded-full overflow-hidden">
                            <div
                                className={`h-full transition-all duration-1000 ease-out ${isHighest ? 'bg-white' : 'bg-white/40'
                                    }`}
                                style={{ width: `${probability * 100}%` }}
                            />
                        </div>
                    </div>
                ))}
            </div>

            {/* Processing Time */}
            {processingTime && (
                <>
                    <div className="w-full h-px bg-white/10 my-6" />
                    <div className="text-center">
                        <p className="text-xs text-white/40">
                            Processed in {processingTime.toFixed(0)}ms
                        </p>
                    </div>
                </>
            )}
        </div>
    );
};

export default ResultsCard;
