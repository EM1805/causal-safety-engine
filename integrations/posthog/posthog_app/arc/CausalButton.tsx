import { useState } from 'react'

type Cause = {
    feature: string
    effect: number
    explanation?: string
    stability?: number
}

type AnalyzeResult = {
    causes: Cause[]
    confidence: number
    suggestion: string
}

type Props = {
    rows: Array<Record<string, unknown>>
    apiBaseUrl: string
}

export function CausalButton({ rows, apiBaseUrl }: Props): JSX.Element {
    const [result, setResult] = useState<AnalyzeResult | null>(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState<string | null>(null)

    const runAnalysis = async (): Promise<void> => {
        setLoading(true)
        setError(null)

        const response = await fetch(`${apiBaseUrl}/analyze`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ rows }),
        })

        if (!response.ok) {
            setError('Analysis failed. Please retry with richer data.')
            setLoading(false)
            return
        }

        const payload: AnalyzeResult = await response.json()
        setResult(payload)
        setLoading(false)
    }

    return (
        <div>
            <button onClick={runAnalysis} disabled={loading}>
                {loading ? 'Running…' : '✨ Explain why'}
            </button>

            {error && <div>{error}</div>}

            {result && (
                <div>
                    <h3>Top causes</h3>
                    {result.causes.map((cause) => (
                        <div key={cause.feature}>
                            <div>{cause.explanation ?? `${cause.feature}: ${cause.effect}`}</div>
                            {cause.stability !== undefined && <div>Stability: {cause.stability}</div>}
                        </div>
                    ))}
                    <div>Confidence: {result.confidence}</div>
                    <div>Suggestion: {result.suggestion}</div>
                </div>
            )}
        </div>
    )
}
