import { useState, useEffect } from "react";

export function useUserTimeZone() {
  const [timeZone, setTimeZone] = useState("UTC");

  useEffect(() => {
    if (typeof window !== "undefined") {
      const tz = Intl.DateTimeFormat().resolvedOptions().timeZone;
      setTimeZone(tz);
    }
  }, []);

  return timeZone;
}
