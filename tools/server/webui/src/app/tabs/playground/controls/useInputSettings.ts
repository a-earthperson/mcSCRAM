import { useCallback, useEffect, useState } from "react";
import { parseZodField } from "../../../lib/parseZodField";
import { isOraclePEnabled, ORACLE_P_OFF, oraclePValueSchema } from "./oracleP";

export type InputSettings = {
  oracleP: number;
  setOracleP: (oracleP: number) => void;
  oracleEnabled: boolean;
};

export type UseInputSettingsOptions = {
  selectedId: string;
  oraclePDefault: number | null;
};

export function useInputSettings({ selectedId, oraclePDefault }: UseInputSettingsOptions): InputSettings {
  const [oracleP, setOraclePState] = useState(ORACLE_P_OFF);

  useEffect(() => {
    if (oraclePDefault === null) {
      setOraclePState(ORACLE_P_OFF);
      return;
    }
    const parsed = parseZodField(oraclePValueSchema, oraclePDefault);
    setOraclePState(parsed.ok ? parsed.value : ORACLE_P_OFF);
  }, [oraclePDefault, selectedId]);

  const setOracleP = useCallback((next: number) => {
    if (!isOraclePEnabled(next)) {
      setOraclePState(ORACLE_P_OFF);
      return;
    }
    const parsed = parseZodField(oraclePValueSchema, next);
    if (parsed.ok) setOraclePState(parsed.value);
  }, []);

  return {
    oracleP,
    setOracleP,
    oracleEnabled: isOraclePEnabled(oracleP),
  };
}
