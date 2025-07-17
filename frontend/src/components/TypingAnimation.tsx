import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'

interface TypingAnimationProps {
    text: string
    speed?: number
    className?: string
    onComplete?: () => void
}

export default function TypingAnimation({
    text,
    speed = 50,
    className = '',
    onComplete
}: TypingAnimationProps) {
    const [displayedText, setDisplayedText] = useState('')
    const [currentIndex, setCurrentIndex] = useState(0)
    const [isComplete, setIsComplete] = useState(false)

    useEffect(() => {
        if (currentIndex < text.length) {
            const timer = setTimeout(() => {
                setDisplayedText(text.slice(0, currentIndex + 1))
                setCurrentIndex(currentIndex + 1)
            }, speed)
            return () => clearTimeout(timer)
        } else if (!isComplete && text.length > 0) {
            setIsComplete(true)
            onComplete?.()
        }
    }, [currentIndex, text, speed, isComplete, onComplete])

    useEffect(() => {
        // Reset when text changes
        setDisplayedText('')
        setCurrentIndex(0)
        setIsComplete(false)
    }, [text])

    const containerVariants = {
        hidden: { opacity: 0 },
        visible: {
            opacity: 1,
            transition: {
                staggerChildren: 0.02
            }
        }
    }

    const characterVariants = {
        hidden: {
            opacity: 0,
            y: 10,
            filter: 'blur(4px)'
        },
        visible: {
            opacity: 1,
            y: 0,
            filter: 'blur(0px)',
            transition: {
                duration: 0.3,
                ease: "easeOut" as const
            }
        }
    }

    return (
        <div className={`relative ${className}`}>
            <motion.span
                variants={containerVariants}
                initial="hidden"
                animate="visible"
                className="inline"
            >
                {displayedText.split('').map((char, index) => (
                    <motion.span
                        key={`${text}-${index}`}
                        variants={characterVariants}
                        className="inline"
                    >
                        {char === ' ' ? '\u00A0' : char}
                    </motion.span>
                ))}
                <motion.span
                    className="typing-cursor inline-block"
                    animate={{ opacity: [1, 0] }}
                    transition={{ duration: 1, repeat: Infinity }}
                />
            </motion.span>
        </div>
    )
} 