# KairosTrack Product Requirements Document

## Product Summary

KairosTrack is a full event-based time and attendance platform for churches, schools, clubs, teams, businesses, and organizations.

The product helps organizations create events, manage people, track attendance, and report attendance history using multiple check-in methods:

- Manual attendance.
- Swipe attendance.
- QR code attendance.
- NFC attendance.
- Facial recognition.
- GeoTracking.

Attendance must always be linked to an event session.

Example:

`Sylvester checked in for Sunday Service on April 21, 2026.`

## Product Goals

- Make attendance easy for small and large organizations.
- Support multiple attendance methods based on plan tier.
- Keep local camera access on the desktop app for private IP cameras.
- Sync attendance records to a hosted backend.
- Allow members to self check in without exposing other member records.
- Provide clear reports by event, date range, branch, person, and method.
- Prepare the product for subscription pricing.

## Target Users

| User Type | Description |
| --- | --- |
| Organization Owner | Creates organization, manages billing, plan, branches, and users. |
| Admin | Manages members, events, attendance sessions, reports, and settings. |
| Staff | Marks attendance and operates check-in workflows. |
| Member | Self-registers or self-checks in when allowed. |

## Core Records

| Record | Purpose |
| --- | --- |
| Organization | Customer account using KairosTrack. |
| Branch / Location | Physical location or department location. |
| Member / Person | Person whose attendance is tracked. |
| Staff User | User who manages the platform. |
| Event | Reason attendance is taken, such as Sunday Service or Staff Shift. |
| Attendance Session | A specific occurrence of an event. |
| Attendance Record | A person's attendance status for one session. |
| Device | Camera, NFC reader, phone, or check-in device. |
| Subscription Plan | Controls product features and limits. |
| Audit Log | Tracks important admin and Enterprise actions. |

## Event-Based Attendance Model

An Event is reusable.

Examples:

- Sunday Service.
- Midweek Meeting.
- Choir Rehearsal.
- Staff Shift.
- Youth Event.
- Training Session.
- Conference Day 1.
- Department Meeting.

An Attendance Session is one occurrence of an event.

Examples:

- Sunday Service - April 21, 2026 - 10:00 AM.
- Staff Shift - April 22, 2026 - Morning.
- Training Session - May 3, 2026 - 2:00 PM.

Every attendance record must belong to an attendance session.

## Plan Tiers

### Starter

For small groups that need basic attendance.

Included:

- One organization.
- Limited members.
- One admin/staff user.
- Member management.
- Event creation.
- Attendance sessions.
- Manual attendance.
- Traditional swipe attendance.
- Basic reports.
- CSV export.

Attendance methods:

- Manual.
- Swipe.

### Standard

For churches, schools, clubs, and teams that want faster check-in.

Included:

- Everything in Starter.
- QR code attendance.
- Member self-registration.
- Event-based dashboards.
- PDF and Excel exports.
- Multiple staff users.
- Basic mobile-friendly check-in.
- Email reports.

Attendance methods:

- Manual.
- Swipe.
- QR code.

QR should support:

- Member scans an event QR to check in.
- Staff scans a member QR to mark attendance.

### Professional

Main paid plan for organizations using hardware and advanced attendance.

Included:

- Everything in Standard.
- NFC attendance.
- Facial recognition attendance.
- Camera setup.
- Local desktop camera app.
- Multi-camera support.
- Advanced reports.
- Department/group filtering.
- Attendance trends.
- Optional late/early tracking.
- Multiple branches/locations.
- Role-based access.

Attendance methods:

- Manual.
- Swipe.
- QR code.
- NFC.
- Facial recognition.

### Enterprise

For larger organizations that need advanced control, mobile workforce tracking, and custom workflows.

Included:

- Everything in Professional.
- GeoTracking mobile app.
- Multi-organization or multi-branch management.
- Custom attendance rules.
- API access.
- Audit logs.
- Advanced permissions.
- Dedicated onboarding.
- Custom branding.
- Priority support.
- Device management.
- Data retention controls.
- Custom integrations.
- Optional SSO in future.

Attendance methods:

- Manual.
- Swipe.
- QR code.
- NFC.
- Facial recognition.
- GeoTracking.

## Attendance Methods

### Manual

Admin or staff selects members and marks them present or absent.

Main platform:

- Desktop.

Use case:

- Small groups.
- Admin-controlled attendance.
- Backup when hardware is unavailable.

### Swipe

Staff reviews members one by one and swipes:

- Right = Present.
- Left = Absent.

Main platform:

- Mobile.

Use case:

- Fast staff-led attendance for classes, events, or groups.

### QR Code

QR attendance supports low-cost check-in.

Modes:

- Event QR: member scans event QR and checks in.
- Member QR: staff scans member QR and marks attendance.

Main platform:

- Mobile.
- Hosted public/member pages.

Use case:

- Churches, schools, events, conferences, clubs.

### NFC

Member taps a card or tag to check in.

Main platform:

- Desktop with reader.
- Mobile later if phone NFC scanning is required.

Use case:

- Staff shifts.
- Schools.
- Controlled access points.

### Facial Recognition

Desktop app uses local cameras and face recognition to mark attendance.

Main platform:

- Desktop app.

Reason:

Hosted backend cannot access private IP cameras such as LAN RTSP streams. Camera access must remain local to the installed desktop machine.

Use case:

- Fixed locations.
- Churches.
- Offices.
- Classrooms.

### GeoTracking

Mobile app checks whether a staff/member is within an approved location radius.

Main platform:

- Mobile.

Use case:

- Field teams.
- Multi-branch organizations.
- Staff shifts.
- Enterprise customers.

Privacy requirement:

- Location permission must be clear.
- User must understand why location is needed.
- Location should only be captured during check-in/check-out.

## Platform Responsibilities

### Backend API

Owns:

- Organizations.
- Branches.
- Staff users.
- Members.
- Events.
- Attendance sessions.
- Attendance records.
- Plans and limits.
- Reports.
- Audit logs.
- Device records.
- Public/member self-registration.
- Authentication and authorization.

### Desktop App

Owns:

- Staff login.
- Local camera streaming.
- Facial recognition.
- Manual attendance.
- NFC reader workflow.
- QR admin workflow.
- Reports and exports.
- Desktop settings.
- Installer and updates.

Desktop must call the backend API for CRUD and attendance records.

### Mobile App

Owns:

- Staff login.
- Staff dashboard.
- Session list.
- Swipe attendance.
- Staff QR scanning.
- Member self check-in.
- Member dashboard.
- GeoTracking.
- Offline/pending sync in future.

## Public Website

The public product page should explain:

- KairosTrack brand.
- Event-based attendance.
- Supported attendance methods.
- Pricing plans.
- Book demo form.
- Download app option.
- Product benefits.

The product page should not expose private admin/member data.

## Security Requirements

- Staff login should use backend username and password.
- Access tokens should refresh automatically when possible.
- Expired sessions should redirect to login.
- Member self-service must not expose other member records.
- Role-based access should protect owner/admin/staff/member workflows.
- Enterprise audit logs should track sensitive actions.
- Production secrets must be stored in environment variables.
- Public endpoints must be limited to safe self-registration and check-in flows.

## Reporting Requirements

Reports should support:

- Event report.
- Session report.
- Member report.
- Date range report.
- Branch report.
- Attendance method breakdown.
- Present/absent/pending/check-out counts.
- CSV export.
- Excel export.
- PDF export.
- Email reports.

Future reports:

- Trends.
- Late/early.
- Department/group comparison.
- GeoTracking compliance.
- Device usage.

## Billing Requirements

Plan enforcement should control:

- Maximum organizations.
- Maximum members.
- Maximum staff users.
- Branch count.
- Event/session features.
- Attendance methods.
- Report/export access.
- Device/camera count.
- API access.
- Audit logs.
- GeoTracking.

Payment integration is not required for initial MVP but should be added before commercial launch.

## MVP Scope

The MVP should include:

- Organization onboarding.
- Staff login.
- Member management.
- Event and attendance session management.
- Manual attendance.
- Mobile swipe attendance.
- QR check-in.
- NFC check-in.
- Facial recognition tied to selected event session.
- GeoTracking foundation.
- Reports and exports.
- Desktop installer.
- Public product page.

## Recommended Build Order

1. Event-based attendance foundation.
2. Manual attendance.
3. Swipe attendance.
4. QR attendance.
5. NFC attendance.
6. Facial recognition tied to event sessions.
7. Mobile member mode.
8. GeoTracking.
9. Subscription limits.
10. Reports and exports.
11. Installer and release pipeline.
12. Production hardening.
13. Billing integration.
14. Enterprise controls.

## Current Completion Estimate

| Area | Estimated Completion |
| --- | --- |
| Backend | 80% |
| Desktop | 82% |
| Mobile | 65% |
| Overall Product | 75% |

## Remaining Work

Backend:

- Automated tests.
- Production security hardening.
- Billing/payment automation.
- Complete audit logs.
- Stronger device management.
- Swagger/API documentation polish.

Desktop:

- Full live end-to-end testing.
- Installer rebuild and test.
- Update notification polish.
- More UI edge-case cleanup.
- Final packaging documentation.

Mobile:

- Staff dashboard polish.
- Member dashboard polish.
- Offline/pending sync.
- NFC phone scanning if required.
- GeoTracking UX refinement.
- App icon and splash screen.
- Release signing and versioning.
- Staff/member workflow tests.

## Acceptance Criteria

- Staff can create/select an event session.
- No attendance record is created without an event session.
- Each attendance method writes the correct method value.
- Reports filter correctly by event, date, branch, and method.
- Desktop face recognition can access private IP cameras locally.
- Mobile member mode does not expose other members.
- Expired sessions are handled cleanly.
- Offline backend errors are friendly.
- Installer uses KairosTrack branding.
- Public product page explains pricing and download/demo paths.

