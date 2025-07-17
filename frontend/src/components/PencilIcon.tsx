import React from 'react';

const PencilIcon = ({ className, style }: { className?: string, style?: React.CSSProperties }) => (
    <svg
        xmlns="http://www.w3.org/2000/svg"
        className={className}
        style={style}
        width="16"
        height="16"
        viewBox="0 0 24 24"
        fill="none"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
    >
        <path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z" />
        <path d="m15 5 4 4" />
    </svg>
);

export default PencilIcon; 