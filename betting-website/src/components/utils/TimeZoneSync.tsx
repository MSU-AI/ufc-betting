"use client";

import { useEffect } from "react";
import { useUserTimeZone } from "@/lib/timeZone";
import { useRouter, usePathname, useSearchParams } from "next/navigation";

export default function TimeZoneSync() {
  const userTimeZone = useUserTimeZone();
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  useEffect(() => {
    if (!userTimeZone) return;
    const params = new URLSearchParams(searchParams?.toString() || "");
    if (!params.has("tz")) {
      params.set("tz", userTimeZone);
      router.replace(`${pathname}?${params.toString()}`);
    }
  }, [userTimeZone, pathname, router, searchParams]);

  return null;
}
