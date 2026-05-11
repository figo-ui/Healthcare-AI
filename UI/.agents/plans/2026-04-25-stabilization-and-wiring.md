# CBHI Suite Stabilization and Wiring Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve localization inconsistencies and broken repository syntax across the Member, Admin, and Facility applications.

**Architecture:** 
- Populate ARB/JSON values for missing keys in all three apps.
- Fix invalid `?variable` syntax in Admin and Facility repositories.
- Add missing appeal endpoints to Member repository.

**Tech Stack:** Flutter (Dart), NestJS (TypeScript)

---

### Task 1: Member App - Localization Fixes

**Files:**
- Modify: `member_based_cbhi/lib/l10n/app_en.arb`
- Modify: `member_based_cbhi/lib/l10n/app_am.arb`
- Modify: `member_based_cbhi/lib/l10n/app_om.arb`

- [ ] **Step 1: Update app_en.arb**
Add missing keys (see previous plan for list).

- [ ] **Step 2: Update app_am.arb**
Add missing Amharic translations.

- [ ] **Step 3: Update app_om.arb**
Add missing Afaan Oromo translations.

- [ ] **Step 4: Commit**
```bash
git add member_based_cbhi/lib/l10n/*.arb
git commit -m "fix(member): complete localization keys"
```

### Task 2: Admin & Facility Apps - Localization Fixes

**Files:**
- Modify: `cbhi_admin_desktop/lib/src/i18n/app_localizations.dart`
- Modify: `cbhi_facility_desktop/lib/src/i18n/app_localizations.dart`

- [ ] **Step 1: Update Admin app_localizations.dart**
Populate `_values` map with missing keys for `en`, `am`, and `om`.

- [ ] **Step 2: Update Facility app_localizations.dart**
Populate `_values` map with missing keys for `en`, `am`, and `om`.

- [ ] **Step 3: Commit**
```bash
git add cbhi_admin_desktop/lib/src/i18n/app_localizations.dart cbhi_facility_desktop/lib/src/i18n/app_localizations.dart
git commit -m "fix(admin,facility): complete hardcoded localizations"
```

### Task 3: Member App - Repository Wiring

**Files:**
- Modify: `member_based_cbhi/lib/src/cbhi_data.dart`

- [ ] **Step 1: Add missing appeal methods**
Implement `getMyAppeals` and `submitClaimAppeal`.

- [ ] **Step 2: Commit**
```bash
git add member_based_cbhi/lib/src/cbhi_data.dart
git commit -m "feat(member): wire claim appeal endpoints"
```

### Task 4: Admin & Facility Apps - Repository Fixes

**Files:**
- Modify: `cbhi_admin_desktop/lib/src/data/admin_repository.dart`
- Modify: `cbhi_facility_desktop/lib/src/data/facility_repository.dart`

- [ ] **Step 1: Fix invalid syntax in AdminRepository**
Replace `'key': ?variable` with `'key': variable` (using ternary if null check is needed).
Example: `'facilityCode': facilityCode` instead of `'facilityCode': ?facilityCode`.

- [ ] **Step 2: Fix invalid syntax in FacilityRepository**
Same fix for `supportingDocumentUpload`.

- [ ] **Step 3: Commit**
```bash
git add cbhi_admin_desktop/lib/src/data/admin_repository.dart cbhi_facility_desktop/lib/src/data/facility_repository.dart
git commit -m "fix(admin,facility): resolve invalid repository syntax"
```

### Task 5: Final Verification

- [ ] **Step 1: Run audit script**
Verify all three apps have 0 missing keys.

- [ ] **Step 2: Dry-run build check**
Ensure repositories compile.

---
