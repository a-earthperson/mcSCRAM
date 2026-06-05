import type { ZodType } from "zod";

export type ParsedField<T> =
  | { ok: true; value: T }
  | { ok: false; message: string };

export function parseZodField<T>(schema: ZodType<T>, raw: unknown): ParsedField<T> {
  const result = schema.safeParse(raw);
  if (result.success) return { ok: true, value: result.data };
  const issue = result.error.issues[0];
  return { ok: false, message: issue?.message ?? "Invalid value" };
}
