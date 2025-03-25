"use client";
import React, { useState, useEffect, useRef, useLayoutEffect } from "react";
import { motion, AnimatePresence, useMotionValue, animate } from "framer-motion";

export interface HeaderAnimationProps {
  animDuration?: number;
  offsetY?: number;
}

export default function HeaderAnimation({ animDuration = 2.5, offsetY = 0 }: HeaderAnimationProps) {
  const [isAnimating, setIsAnimating] = useState(false);
  const [isAnimatingForHoop, setAnimatingForHoop] = useState(false);
  const [ballInHoop, setBallInHoop] = useState(false);
  const progress = useMotionValue(0);
  const ballX = useMotionValue(0);
  const ballY = useMotionValue(0);
  const swirlPathRef = useRef<SVGPathElement | null>(null);
  const [totalLength, setTotalLength] = useState(0);

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsAnimating(true);
      setTimeout(() => setAnimatingForHoop(true), 500);
      setTimeout(() => setBallInHoop(true), 2200);
      setTimeout(() => setAnimatingForHoop(false), 4000);
      setTimeout(() => {
        setIsAnimating(false);
        setBallInHoop(false);
      }, 7500);
    }, 500);
    return () => clearTimeout(timer);
  }, []);

  useLayoutEffect(() => {
    const timer = setTimeout(() => {
      if (swirlPathRef.current) {
        try {
          const length = swirlPathRef.current.getTotalLength();
          setTotalLength(length);
          const startPoint = swirlPathRef.current.getPointAtLength(0);
          ballX.set(startPoint.x - 15);
          ballY.set(startPoint.y - 8);
        } catch (e) {
          console.error("Error measuring path:", e);
        }
      }
    }, 100);
    return () => clearTimeout(timer);
  }, [isAnimatingForHoop]);

  useEffect(() => {
    if (totalLength > 0 && isAnimatingForHoop) {
      progress.set(0);
      const controls = animate(progress, 1, {
        duration: animDuration - 0.35,
        ease: "easeInOut",
        onUpdate: (latest) => {
          if (swirlPathRef.current) {
            try {
              const point = swirlPathRef.current.getPointAtLength(latest * totalLength);
              ballX.set(point.x - 8);
              ballY.set(point.y - 8);
            } catch (e) {
              console.error("Error updating ball position:", e);
            }
          }
        },
      });
      return controls.stop;
    }
  }, [totalLength, isAnimatingForHoop, progress, animDuration]);

  return (
    // Apply the offset here so the entire animation block moves vertically.
    <div className="relative mt-4" style={{ transform: `translateY(${offsetY}px)` }}>
      <AnimatePresence>
        {isAnimatingForHoop && (
          <div className="absolute inset-0 w-full h-full overflow-visible">
            {/* Hoop Animation */}
            <motion.div
              className="absolute top-[15px] left-[-20px] z-10"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0, transition: { duration: 0.3 } }}
            >
              <svg width="50" height="50" viewBox="0 0 50 50">
                {/* Backboard */}
                <motion.rect
                  x="0"
                  y="0"
                  width="5"
                  height="35"
                  fill="white"
                  strokeWidth="1.5"
                  stroke="#333"
                  animate={{
                    x: [0, -2, 2, 0],
                    transition: {
                      delay: animDuration - 0.3,
                      duration: 0.3,
                      times: [0, 0.3, 0.6, 1],
                    },
                  }}
                />
                {/* Rim */}
                <motion.rect
                  x="5"
                  y="17"
                  width="22"
                  height="3"
                  fill="white"
                  strokeWidth="1.5"
                  stroke="#333"
                  animate={{
                    y: [17, 18, 16, 17],
                    transition: {
                      delay: animDuration - 0.3,
                      duration: 0.3,
                      times: [0, 0.3, 0.7, 1],
                    },
                  }}
                />
                {/* Net attachment points */}
                <rect
                  x="5"
                  y="20"
                  width="2"
                  height="2"
                  fill="white"
                  stroke="#333"
                  strokeWidth="0.5"
                />
                <rect
                  x="25"
                  y="20"
                  width="2"
                  height="2"
                  fill="white"
                  stroke="#333"
                  strokeWidth="0.5"
                />
                {/* Net */}
                <motion.path
                  d="M5,22 C5,22 15,22 15,22 C15,22 25,22 25,22 M5,22 L5,30 M15,22 L15,32 M25,22 L25,30"
                  fill="transparent"
                  stroke="white"
                  strokeWidth="1"
                  strokeDasharray="1,1"
                  animate={{
                    d: [
                      "M5,22 C5,22 15,22 15,22 C15,22 25,22 25,22 M5,22 L5,30 M15,22 L15,32 M25,22 L25,30",
                      "M5,22 C5,27 15,30 15,30 C15,30 25,27 25,22 M5,22 L5,30 M15,22 L15,35 M25,22 L25,30",
                      "M5,22 C5,25 15,27 15,27 C15,27 25,25 25,22 M5,22 L5,30 M15,22 L15,33 M25,22 L25,30",
                    ],
                    transition: {
                      delay: animDuration - 0.55,
                      duration: 0.5,
                      times: [0, 0.7, 1],
                    },
                  }}
                />
              </svg>
            </motion.div>

            {/* Swirl and Ball Animation */}
            <div className="absolute z-0 w-56 h-20 top-0 left-0 transform -translate-x-6">
              <svg
                className="w-full h-full pointer-events-none"
                viewBox="0 0 180 80"
                style={{ pointerEvents: "none" }}
              >
                {/* Swirl Path */}
                <motion.path
                  ref={swirlPathRef}
                  d="M180,40 C160,10 140,70 120,40 S60,10 0,40"
                  fill="transparent"
                  stroke="white"
                  strokeWidth="2"
                  strokeDasharray="5,5"
                  initial={{ pathLength: 0, opacity: 0 }}
                  animate={{ pathLength: 1, opacity: [0, 1, 1, 0] }}
                  transition={{
                    pathLength: { duration: animDuration, ease: "easeInOut" },
                    opacity: {
                      duration: animDuration + 1,
                      times: [0, 0.1, 0.8, 1],
                    },
                  }}
                />
                {totalLength > 0 && (
                  <motion.g
                    style={{ x: ballX, y: ballY }}
                    initial={{ opacity: 0, rotate: 0 }}
                    animate={{ opacity: [0, 1, 1, 1, 0], rotate: 360 }}
                    transition={{
                      duration: animDuration + 1,
                      ease: "easeInOut",
                      opacity: {
                        duration: animDuration + 0.875,
                        times: [0, 0.1, 0.9, 0.98, 1],
                      },
                    }}
                  >
                    <circle
                      cx="8"
                      cy="8"
                      r="8"
                      fill="#FF7900"
                      stroke="#333"
                      strokeWidth="1.5"
                    />
                    <path d="M0,8 L16,8 M8,0 L8,16" stroke="#333" strokeWidth="1.5" />
                    <path
                      d="M2,2 C5,5 11,5 14,2 M2,14 C5,11 11,11 14,14"
                      fill="transparent"
                      stroke="#333"
                      strokeWidth="1.5"
                    />
                  </motion.g>
                )}
              </svg>
            </div>

            {/* Dunk Effect */}
            <motion.div
              className="absolute top-[15px] left-[-38px] w-16 h-16 flex items-center justify-center z-30"
              initial={{ opacity: 0, scale: 0 }}
              animate={{
                opacity: [0, 0.8, 0],
                scale: [0.5, 1.5, 3],
                transition: {
                  delay: animDuration - 0.23,
                  duration: 0.5,
                  times: [0, 0.3, 1],
                },
              }}
            >
              <div className="w-10 h-10 rounded-full bg-white opacity-70"></div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}
