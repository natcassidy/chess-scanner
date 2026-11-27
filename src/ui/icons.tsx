import type { JSX } from 'preact';

type P = JSX.SVGAttributes<SVGSVGElement>;
const base = (children: JSX.Element | JSX.Element[], p: P) => (
  <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" stroke-width="2"
    stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" {...p}>
    {children}
  </svg>
);

export const IconCamera = (p: P) =>
  base(
    [
      <path d="M4 8h3l2-3h6l2 3h3a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V9a1 1 0 0 1 1-1z" />,
      <circle cx="12" cy="13.5" r="3.5" />,
    ],
    p,
  );
export const IconFirst = (p: P) => base([<path d="M18 6l-6 6 6 6" />, <path d="M7 6v12" />], p);
export const IconPrev = (p: P) => base(<path d="M15 6l-6 6 6 6" />, p);
export const IconNext = (p: P) => base(<path d="M9 6l6 6-6 6" />, p);
export const IconLast = (p: P) => base([<path d="M6 6l6 6-6 6" />, <path d="M17 6v12" />], p);
export const IconMore = (p: P) =>
  base(
    [<circle cx="5" cy="12" r="1.2" fill="currentColor" />, <circle cx="12" cy="12" r="1.2" fill="currentColor" />,
      <circle cx="19" cy="12" r="1.2" fill="currentColor" />],
    p,
  );
export const IconClose = (p: P) => base([<path d="M6 6l12 12" />, <path d="M18 6L6 18" />], p);
export const IconImage = (p: P) =>
  base(
    [<rect x="3" y="4" width="18" height="16" rx="2" />, <circle cx="9" cy="10" r="2" />,
      <path d="M21 16l-5-5-9 9" />],
    p,
  );
export const IconHistory = (p: P) =>
  base([<path d="M3 12a9 9 0 1 0 3-6.7" />, <path d="M3 4v4h4" />, <path d="M12 8v4l3 2" />], p);
export const IconFlip = (p: P) =>
  base([<path d="M7 4v16" />, <path d="M4 7l3-3 3 3" />, <path d="M17 20V4" />, <path d="M14 17l3 3 3-3" />], p);
export const IconEdit = (p: P) =>
  base([<path d="M4 20h4L19 9l-4-4L4 16v4z" />, <path d="M14 6l4 4" />], p);
export const IconTrash = (p: P) =>
  base([<path d="M4 7h16" />, <path d="M10 11v6M14 11v6" />, <path d="M6 7l1 13h10l1-13" />, <path d="M9 7V4h6v3" />], p);
export const IconReset = (p: P) => base([<path d="M4 12a8 8 0 1 0 2.3-5.6" />, <path d="M4 4v5h5" />], p);
export const IconCopy = (p: P) =>
  base([<rect x="8" y="8" width="12" height="12" rx="2" />, <path d="M16 8V5a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h3" />], p);
export const IconExternal = (p: P) =>
  base([<path d="M14 4h6v6" />, <path d="M20 4l-9 9" />, <path d="M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5" />], p);
export const IconStar = (p: P) => base(<path d="M12 4l2.5 5 5.5.8-4 3.9.9 5.5L12 16.6 7.1 19.2l.9-5.5-4-3.9 5.5-.8z" />, p);
export const IconRotate = (p: P) =>
  base([<path d="M20 12a8 8 0 1 1-2.3-5.6" />, <path d="M20 4v5h-5" />], p);
export const IconCheck = (p: P) => base(<path d="M5 12l5 5 9-10" />, p);
export const IconEraser = (p: P) =>
  base([<path d="M7 20h10" />, <path d="M5 15l8-8 6 6-5 5H8z" />], p);
